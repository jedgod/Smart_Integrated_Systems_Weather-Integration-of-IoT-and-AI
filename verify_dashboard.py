from app import app

with app.test_client() as c:
    r = c.get('/')
    html = r.get_data(as_text=True)
    print('STATUS', r.status_code)
    print('HAS_TEMPLATE_ERROR', ('TemplateSyntaxError' in html) or ('expected token' in html))
    print('HAS_DASHBOARD', 'id="dashboard"' in html)
    print('HAS_WEATHER', 'id="weather"' in html)
    print('HAS_PREDICTION', 'id="prediction"' in html)
    print('HAS_EVALUATION', 'id="evaluation"' in html)
    print('HAS_HISTORICAL', 'id="historical"' in html)
    print('HAS_SETTINGS', 'id="settings"' in html)
    print('HAS_ABOUT', 'id="about"' in html)
    print('HAS_CITY_SELECTOR', 'id="citySelect"' in html)
    print('HAS_BRAND', ('JERRY DIABOR' in html) and ('SMART INTEGRATED SYSTEMS' in html))
    print('HAS_NAV', ('data-section="dashboard"' in html) and ('data-section="weather"' in html) and ('data-section="prediction"' in html))
    print('HAS_JS_NAV', 'scrollIntoView' in html)
    print('HAS_VALID_JINJA', "{% if city == 'New York' %}" in html)
