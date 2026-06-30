from runpod_runner.pod_manager import PodManager


def test_extract_ssh_endpoint_from_runtime_ports_list():
    pod = {
        "runtime": {
            "ports": [
                {
                    "ip": "203.0.113.10",
                    "isIpPublic": True,
                    "privatePort": 22,
                    "publicPort": 14022,
                    "type": "tcp",
                }
            ]
        }
    }

    assert PodManager._extract_ssh_endpoint(pod) == ("203.0.113.10", 14022)


def test_extract_ssh_endpoint_from_port_mappings_dict():
    pod = {
        "runtime": {
            "publicIp": "203.0.113.11",
            "portMappings": {"22": 14023},
        }
    }

    assert PodManager._extract_ssh_endpoint(pod) == ("203.0.113.11", 14023)
