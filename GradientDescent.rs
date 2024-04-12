fn main() {
    // feature values
    let feature_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let x: Vec<f64> = feature_int.iter().map(|&xi| xi as f64).collect();
    
    // target values
    let target_int = [3, 5, 7, 9, 13, 13, 17, 19, 21, 25, 27, 29]; 
    let y: Vec<f64> = target_int.iter().map(|&yi| yi as f64).collect();
    
    let n = y.len() as f64; // number of samples

    // initialisation of learnable parameters
    let mut m = 0.0; // slope
    let mut c = 0.0; // y-intercept
    
    // initialisation of variables
    let mut iteration = 0;
    let mut _mse = 0.0; // mean squared error
    // '_var' to supress 'unused_assignments' warning
    
    // gradient descent parameters
    let lr = 0.01; // learning rate
    let max_iters = 100; // maximum iterations (or) epochs

    loop {
        // y_pred = m * x + c
        let y_pred_tmp: Vec<f64> = x.iter().map(|&f| f as f64 * m).collect();
        let y_pred: Vec<f64> = y_pred_tmp.iter().map(|&yp| yp + c).collect();

        // gradient_m = (2 / n) * sum((y_pred - y) * x)
        let diff: Vec<f64> = y_pred
            .iter()
            .zip(&y)
            .map(|(&yp, &yi)| yp - yi)
            .collect();
            
        let multiplied_diff: Vec<f64> = diff
            .iter()
            .zip(&x)
            .map(|(&diff, &x)| diff * x as f64)
            .collect();
            
        let sum: f64 = multiplied_diff
            .iter()
            .sum();
            
        let gradient_m = (2.0 / n) * sum;

        // gradient_c = (2 / n) * sum(y_pred - y) ['n' to normalise the value]
        let sum: f64 = diff
            .iter()
            .sum();
            
        let gradient_c = (2.0 / n) * sum;

        // update parameters using gradient descent algorithm
        m = m - lr * gradient_m;
        c = c - lr * gradient_c;

        // error = (y_pred - y).pow(2)
        let errors: Vec<f64> = y_pred
        .iter()
        .zip(&y)
        .map(|(&yp, &yi)| (yp - yi).powi(2))
        .collect();
        
        _mse = errors.iter().sum();
        
        // termination condition
        if iteration >= max_iters {
            break;
        }
        
        iteration += 1;
    }

    // output
    println!("Epochs: {}", max_iters);
    println!("Learning Rate: {}", lr);
    println!("Slope: {}", m);
    println!("Intercept: {}", c);
    println!("Mean Squared Error: {}", _mse);
}
