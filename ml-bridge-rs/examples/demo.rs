use eerie::runtime::api;
use eerie::runtime::hal;

use ml_bridge_rs::load_exported_fn;

load_exported_fn!("../ml-bridge-py/test_outputs/export_data.json");
load_exported_fn!("../ml-bridge-py/test_outputs/fn_with_external.json");
load_exported_fn!("../ml-bridge-py/test_outputs/fn_with_scalars.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = api::Instance::new(
        &api::InstanceOptions::new(&mut hal::DriverRegistry::new()).use_all_available_drivers(),
    )
    .unwrap();
    let device = instance
        .try_create_default_device("local-sync")
        .expect("Failed to create device");
    let session =
        api::Session::create_with_device(&instance, &api::SessionOptions::default(), &device)
            .unwrap();

    let inp1 = [1f32; 128];
    let inp2 = [[1f32; 128]; 128];
    let inp3 = [1f32; 128];
    let inp4 = [[1f32; 128]; 128];
    let inp5 = [1f32; 128];

    test_fn::setup_function(&instance, &session);
    {
        let start = std::time::Instant::now();

        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));
        let out = test_fn::test_fn(&instance, &session, (&inp1, &inp2, &inp3, &inp4, &inp5));

        let duration = start.elapsed();
        println!("Execution took: {:?}", duration / 10);
        println!("out:\t{:?}", out);
    }

    Ok(())
}
