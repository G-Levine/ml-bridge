use eerie::runtime::api;
use eerie::runtime::hal;

use ml_bridge_rs::make_fn_from_json;

make_fn_from_json!("../ml-bridge-py/export_data.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = api::Instance::new(
        &api::InstanceOptions::new(&mut hal::DriverRegistry::new()).use_all_available_drivers(),
    )
    .unwrap();
    let device = instance
        .try_create_default_device("local-task")
        .expect("Failed to create device");
    let session =
        api::Session::create_with_device(&instance, &api::SessionOptions::default(), &device)
            .unwrap();

    let inp1: [f32; 3] = [1.0, 2.0, 3.0];
    let inp2: [f32; 3] = [4.0, 5.0, 6.0];
    let inp3: [[f32; 3]; 3] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

    {
        let start = std::time::Instant::now();

        let out = test_fn(&instance, &session, (&inp1, &inp2, &inp3));

        let duration = start.elapsed();
        println!("Execution took: {:?}", duration);
        println!("out:\t{:?}", out);
    }

    Ok(())
}
