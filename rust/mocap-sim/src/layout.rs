//! Data-driven room, camera ring and demonstration choreography.
use crate::{demo, math::{v3, V3}, rig};
use serde_json::Value;

pub fn vector(value: &Value) -> V3 {
    let a = value.as_array().expect("expected 3-D vector array");
    assert_eq!(a.len(), 3, "vector must have three components");
    v3(a[0].as_f64().expect("x"), a[1].as_f64().expect("y"), a[2].as_f64().expect("z"))
}

pub fn flights(config: &Value) -> Vec<demo::FlightSpec> {
    config["flights"].as_array().expect("layout flights array")
        .iter().enumerate().map(|(i, f)| {
            assert_eq!(f["body_id"].as_u64(), Some(i as u64), "flight body order must match body_id");
            let scale = f["scale"].as_f64().expect("flight scale");
            let frequency_scale = f["frequency_scale"].as_f64().unwrap_or(1.0);
            assert!(scale > 0.0 && frequency_scale > 0.0, "flight scales must be positive");
            demo::FlightSpec { pad: vector(&f["pad"]), center: vector(&f["center"]),
                scale, phase: f["phase"].as_f64().expect("flight phase"), frequency_scale }
        }).collect()
}

pub fn cameras(config: &Value) -> Vec<rig::Camera> {
    let ring = &config["camera_ring"];
    let height = ring["height_m"].as_f64().expect("camera height_m");
    assert!(height <= 5.0, "compact camera installation must not exceed 5 m");
    rig::ring_rig(ring["radius_m"].as_f64().expect("camera radius_m"), height, vector(&ring["target"]))
}

pub fn apply_metadata(dir: &str, config: &Value) {
    let file = format!("{dir}/rig_blender.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    value["volume"] = config["volume"].clone();
    value["hangar"] = config["hangar"].clone();
    value["layout_name"] = config["name"].clone();
    std::fs::write(&file, serde_json::to_string_pretty(&value).unwrap()).unwrap();
}
