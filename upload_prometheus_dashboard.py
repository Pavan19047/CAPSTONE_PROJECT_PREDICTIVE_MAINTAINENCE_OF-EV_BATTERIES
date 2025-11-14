import json
import requests
from requests.auth import HTTPBasicAuth

# Grafana configuration
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"

# Read the dashboard JSON
with open('grafana_dashboard_prometheus.json', 'r') as f:
    dashboard_data = json.load(f)

# Get Prometheus datasource UID
response = requests.get(
    f"{GRAFANA_URL}/api/datasources/name/Prometheus",
    auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD)
)

if response.status_code == 200:
    prometheus_uid = response.json()['uid']
    print(f"Found Prometheus datasource with UID: {prometheus_uid}")
    
    # Replace all datasource references
    dashboard_json = json.dumps(dashboard_data)
    dashboard_json = dashboard_json.replace('"uid": "prometheus"', f'"uid": "{prometheus_uid}"')
    dashboard_data = json.loads(dashboard_json)
    
    # Wrap dashboard in required format
    payload = {
        "dashboard": dashboard_data,
        "overwrite": True,
        "message": "Updated with Prometheus datasource"
    }
    
    # Upload to Grafana
    upload_response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        json=payload,
        auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD),
        headers={"Content-Type": "application/json"}
    )
    
    if upload_response.status_code == 200:
        result = upload_response.json()
        dashboard_url = f"{GRAFANA_URL}{result['url']}"
        print(f"✅ Dashboard uploaded successfully!")
        print(f"Dashboard URL: {dashboard_url}")
        print(f"Dashboard UID: {result['uid']}")
    else:
        print(f"❌ Failed to upload dashboard: {upload_response.status_code}")
        print(upload_response.text)
else:
    print(f"❌ Failed to get Prometheus datasource: {response.status_code}")
    print(response.text)
