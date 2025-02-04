use eerie::runtime::api;
use eerie::runtime::hal;

use ml_bridge_rs::load_exported_fn;

load_exported_fn!(my_fn, "../ml-bridge-py/test_outputs/my_fn.json");
load_exported_fn!(fn_b, "../ml-bridge-py/test_outputs/another_fn.json");
load_exported_fn!(fn_c, "../ml-bridge-py/test_outputs/fn_with_external.json");
load_exported_fn!(fn_d, "../ml-bridge-py/test_outputs/fn_with_scalars.json");

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

    // Benchmark the execution time
    let num_iterations = 10;
    my_fn::register(&instance, &session);
    {
        let start = std::time::Instant::now();

        let mut out = my_fn::call(&instance, &session, (&inp1, &inp2, &inp3));
        for _ in 1..num_iterations {
            out = my_fn::call(&instance, &session, (&inp1, &inp2, &inp3));
        }

        let duration = start.elapsed();
        println!("out:\t{:?}\n", out);
        println!("Execution took: {:?}", duration / num_iterations);
    }

    Ok(())
}
