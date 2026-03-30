from flask import Flask, request, render_template, Blueprint
import pickle
import pandas as pd
import os
import numpy as np
import json
import requests
try:
    import requests_cache
except Exception:
    requests_cache = None

app = Flask(__name__, template_folder='../templates', static_folder='../static')
crop_recommendation_app = Blueprint('crop_recommendation_app', __name__)

# Resolve project Data directory robustly (project root / Data)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'Data')
model_path = os.path.join(data_dir, 'crop_model.pkl')
le_path = os.path.join(data_dir, 'label_encoder.pkl')
csv_path = os.path.join(data_dir, 'Crop_Recommendation.csv')

crop_model = None
label_encoder = None


def load_crop_model():
    """Load crop recommendation model and label encoder on first use."""
    global crop_model, label_encoder
    if crop_model is not None and label_encoder is not None:
        return True
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                crop_model = pickle.load(f)
        if os.path.exists(le_path):
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
        if crop_model is None or label_encoder is None:
            print('Crop model or label encoder not found in Data/.')
            return False
        return True
    except Exception as e:
        print(f"Error loading crop model or label encoder lazily: {e}")
        return False
feature_stats = {}

try:
    with open(model_path, 'rb') as f:
        crop_model = pickle.load(f)

    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or label encoder: {e}")

def load_feature_stats():
    try:
        df = pd.read_csv(csv_path)
        cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
        df = df[cols]
        stats = {}
        for c in cols:
            col = df[c].dropna().astype(float)
            stats[c] = {
                'min': float(col.min()),
                'max': float(col.max()),
                'p1': float(col.quantile(0.01)),
                'p99': float(col.quantile(0.99)),
                'mean': float(col.mean()),
                'std': float(col.std())
            }
        return stats
    except Exception as e:
        print(f"Error loading feature stats: {e}")
        return {}

feature_stats = load_feature_stats()


def load_state_crop_map():
    """Load state-to-crops mapping used by location/state suggestions."""
    mapping_path = os.path.join(os.path.dirname(__file__), 'state_to_crops.json')
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state mapping: {e}")
        return {}

# Domain overrides for realistic/Indian environment limits (used to broaden dataset percentiles)
domain_limits = {
    # allow higher temperatures frequently seen in India
    'Temperature': {'min': 5.0, 'max': 50.0},
    # humidity as percentage
    'Humidity': {'min': 10.0, 'max': 100.0},
    # pH typical agricultural soils
    'pH_Value': {'min': 3.0, 'max': 10.0},
    # rainfall in mm - allow heavier extremes in some regions
    'Rainfall': {'min': 0.0, 'max': 400.0}
}


def validate_input(values):
    """Validate numeric inputs against dataset-based percentiles (1st-99th)
    combined with domain overrides. Returns (True, None) if all OK,
    otherwise (False, message) where message describes which fields are out-of-range
    using the combined bounds.
    """
    bad = []
    for k, v in values.items():
        s = feature_stats.get(k)
        dom = domain_limits.get(k, {})
        # determine allowed low/high by combining dataset percentiles and domain overrides
        if s:
            low = s.get('p1', float('-inf'))
            high = s.get('p99', float('inf'))
        else:
            low = float('-inf')
            high = float('inf')
        if 'min' in dom:
            # allow domain min to extend below dataset lower percentile
            low = min(low, float(dom['min']))
        if 'max' in dom:
            # allow domain max to extend above dataset upper percentile
            high = max(high, float(dom['max']))
        # if still infinite, skip validation for this feature
        if low == float('-inf') or high == float('inf'):
            continue
        if v < low or v > high:
            bad.append((k, v, low, high))
    if bad:
        msgs = [f"{k}={v} (expected ≈[{low:.2f}-{high:.2f}])" for (k, v, low, high) in bad]
        return False, "Out-of-range values: " + "; ".join(msgs)
    return True, None


# Mahalanobis OOD check: load precomputed mean/inv_cov/threshold if available
maha_path = os.path.join(data_dir, 'mahalanobis.npz')
maha_data = None
maha_mu = None
maha_inv_cov = None
maha_threshold = None
if os.path.exists(maha_path):
    try:
        maha_data = np.load(maha_path)
        maha_mu = maha_data['mu']
        maha_inv_cov = maha_data['inv_cov']
        maha_threshold = float(maha_data['threshold'])
    except Exception as e:
        print(f"Error loading Mahalanobis data: {e}")


def mahalanobis_distance(vec):
    """Compute Mahalanobis distance for a 1D array-like matching feature order."""
    if maha_mu is None or maha_inv_cov is None:
        return None
    x = np.asarray([vec[c] for c in ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall']], dtype=float)
    delta = x - maha_mu
    d2 = float(delta.dot(maha_inv_cov).dot(delta.T))
    return d2


def match_state_to_crops(address, state_map):
    """Match a geocoder `address` dict to a state key in `state_map`.
    Returns tuple: (resolved_state_string_or_None, crops_list_or_empty, matched_key_or_None, debug_msgs:list)
    This is extracted for unit-testing and to keep matching logic isolated.
    """
    debug = []
    state = address.get('state') or address.get('region') or address.get('county')

    def _normalize(s):
        if not s:
            return None
        s = s.strip()
        for token in [' state', ' state of', ' district', ' district of', ' province', ' union territory', ' nct of', ' national capital territory of']:
            if s.lower().endswith(token):
                s = s[:-len(token)].strip()
        s = ' '.join(s.split())
        return s

    alias_map = {
        'orissa': 'Odisha',
        'pondicherry': 'Puducherry',
        'nct of delhi': 'Delhi',
        'national capital territory of delhi': 'Delhi',
        'delhi': 'Delhi',
        'andaman & nicobar islands': 'Andaman and Nicobar Islands'
    }

    def apply_alias(s):
        if not s:
            return s
        return alias_map.get(s.lower(), s)

    candidates = []
    if state:
        candidates.extend([state, _normalize(state), apply_alias(state)])
    for field in ('state_district', 'county', 'region', 'city', 'town', 'village', 'municipality'):
        v = address.get(field)
        if v:
            candidates.extend([v, _normalize(v), apply_alias(v)])

    norm_map = {k: v for k, v in state_map.items()}
    crops = []
    matched_key = None

    for cand in candidates:
        if not cand:
            continue
        if cand in norm_map:
            matched_key = cand
            crops = norm_map[cand]
            debug.append(f"exact candidate match: {cand!r} -> {matched_key!r}")
            break
        cand2 = apply_alias(_normalize(cand))
        if cand2 and cand2 in norm_map:
            matched_key = cand2
            crops = norm_map[cand2]
            debug.append(f"alias candidate match: {cand!r} -> {cand2!r}")
            break

    if not crops and state:
        s_low = state.lower()
        for k, v in norm_map.items():
            if (k.lower() in s_low) or (s_low in k.lower()):
                matched_key = k
                crops = v
                debug.append(f"substring match: state {state!r} matched key {k!r}")
                break
        if not crops:
            for k, v in norm_map.items():
                if any((cand and cand.lower() in k.lower()) for cand in candidates if cand):
                    matched_key = k
                    crops = v
                    debug.append(f"candidate contained in mapping key: matched {k!r}")
                    break

    if not state:
        debug.append(f"no state field in reverse-geocode address; address contents: {address}")

    return state, crops, matched_key, debug


@crop_recommendation_app.route('/')
def home():
    # pass domain limits to template for client-side validation and location UI
    state_options = sorted(load_state_crop_map().keys(), key=str.lower)
    return render_template('recommend.html', domain_limits={
        'Nitrogen': [0,140], 'Phosphorus': [5,145], 'Potassium': [5,205],
        'Temperature': [5,50], 'Humidity': [10,100], 'pH_Value': [3,10], 'Rainfall': [0,400]
    }, state_options=state_options)


@crop_recommendation_app.route('/state-suggest', methods=['POST'])
def state_suggest():
    """Return mapped crop suggestions for a selected state."""
    state = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        state = payload.get('state')
    if not state:
        state = request.form.get('state')
    if not state:
        return {"error": "Missing state"}, 400

    state_map = load_state_crop_map()
    if not state_map:
        return {"error": "State mapping unavailable"}, 500

    selected_state = str(state).strip()
    crops = state_map.get(selected_state, [])
    matched_key = selected_state

    if not crops:
        _, crops, matched_key, _ = match_state_to_crops({'state': selected_state}, state_map)

    if not crops:
        return {"state": selected_state, "crops": [], "note": "No mapped crops for this state"}

    return {"state": matched_key, "crops": crops}


@crop_recommendation_app.route('/location-suggest', methods=['POST'])
def location_suggest():
    """Given lat/lon in JSON body, reverse-geocode to India state and return suggested crops.
    Expects JSON: {"lat": float, "lon": float}
    """
    try:
        data = request.get_json(force=True)
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
    except Exception:
        return {"error": "Invalid JSON payload, expected {lat, lon}"}, 400

    # Use Nominatim (OpenStreetMap) reverse geocoding for simplicity (rate-limited)
    try:
        # round coordinates to reduce cache keys (approx ~100m granularity)
        lat_r = round(lat, 3)
        lon_r = round(lon, 3)
        # enable requests-cache if available (installed at module import)
        if requests_cache is not None:
            try:
                cache_path = os.path.join(base_dir, 'Data', 'geocode_cache')
                # expire_after=86400 seconds (24h)
                requests_cache.install_cache(cache_path, backend='sqlite', expire_after=86400)
            except Exception as _:
                pass

        r = requests.get('https://nominatim.openstreetmap.org/reverse', params={
            'lat': lat_r, 'lon': lon_r, 'format': 'json', 'zoom': 5, 'addressdetails': 1
        }, headers={'User-Agent': 'EPICS-Agent/1.0'}, timeout=5)
        r.raise_for_status()
        payload = r.json()
        address = payload.get('address', {})
        state = address.get('state') or address.get('region') or address.get('county')
        # detect if response came from requests-cache
        cached_flag = bool(getattr(r, 'from_cache', False))
    except Exception as e:
        return {"error": f"Reverse geocode failed: {e}"}, 502

    # only support India for now
    country = address.get('country') if 'address' in locals() else None
    if country and 'India' not in country:
        return {"error": "Location outside India is not supported"}, 400

    # load mapping and return suggestions
    state_map = load_state_crop_map()

    # Use centralized matching helper so tests can import it
    state, crops, matched_key, debug_msgs = match_state_to_crops(address, state_map)
    for m in debug_msgs:
        print('DEBUG:', m)

    if not crops:
        print(f"DEBUG: no crops mapped for resolved state={state!r}; available keys={list(state_map.keys())}")
        return {"state": state, "crops": [], "note": "No mapped crops for this state", "cached": cached_flag}
    return {"state": state, "crops": crops, "cached": cached_flag}


@crop_recommendation_app.route('/recommend', methods=['POST'])
def recommend():
    try:
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_value = float(request.form['ph_value'])
        rainfall = float(request.form['rainfall'])

        input_values = {
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Temperature': temperature,
            'Humidity': humidity,
            'pH_Value': ph_value,
            'Rainfall': rainfall
        }

        # Validate inputs against dataset-derived bounds
        valid, msg = validate_input(input_values)
        if not valid:
            return render_template('recommend.html', error=msg)

        # Mahalanobis OOD check (if precomputed data exists)
        # Skip strict Mahalanobis check for inputs that are within domain_limits
        try:
            # determine if any input is outside domain overrides — only then apply Mahalanobis
            any_outside_domain = False
            for k, v in input_values.items():
                dom = domain_limits.get(k)
                if dom:
                    if v < dom.get('min', float('-inf')) or v > dom.get('max', float('inf')):
                        any_outside_domain = True
                        break

            if any_outside_domain:
                npz = os.path.join(data_dir, 'mahalanobis.npz')
                if os.path.exists(npz):
                    d = np.load(npz)
                    mu = d['mu']
                    inv_cov = d['inv_cov']
                    threshold = float(d['threshold'])
                    x = np.array([input_values[c] for c in ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall']])
                    diff = x - mu
                    m2 = float(diff.dot(inv_cov).dot(diff))
                    if m2 > threshold:
                        return render_template('recommend.html', error=f"Input appears out-of-distribution (Mahalanobis={m2:.2f} &gt; {threshold:.2f}). Please adjust inputs.")
        except Exception as e:
            print('Mahalanobis check failed:', e)

        input_data = pd.DataFrame([input_values])

        # ensure models are loaded lazily
        if not load_crop_model():
            return render_template('recommend.html', error="Model not loaded. Please train models first.")

        prediction_encoded = crop_model.predict(input_data)[0]
        crop_prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        return render_template('recommendation.html', recommendation=crop_prediction)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('recommend.html', error="Error during prediction. Please check your inputs.")


app.register_blueprint(crop_recommendation_app)