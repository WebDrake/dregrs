extern crate rand;

mod rating;

use rating::Rating;

struct YZLM {
    convergence: f64,
    exponent: f64,
    min_divergence: f64,
}

impl YZLM {
    fn calculate_reputation(&self,
        iterations: &mut usize, diff: &mut f64,
        object_reputation: &mut Vec<f64>,
        user_reputation: &mut Vec<f64>, user_links: &Vec<usize>,
        user_divergence: &mut Vec<f64>, reputation_buf: &mut Vec<f64>,
        ratings: &Vec<Rating>) {
        *iterations = 0;

        calculate_object_reputation(object_reputation,
            user_reputation, ratings);

        loop {
            calculate_user_divergence(user_divergence,
                object_reputation, ratings);

            calculate_user_reputation(user_reputation,
                user_divergence, user_links,
                self.exponent, self.min_divergence);

            reputation_buf.clone_from(object_reputation);

            calculate_object_reputation(object_reputation,
                user_reputation, ratings);

            *diff = 0.0;

            for (o, rep) in object_reputation.iter().enumerate() {
                let aux = rep - reputation_buf[o];
                *diff += aux * aux;
            }

            *iterations += 1;

            if *diff <= self.convergence { break; }
        }
    }
}

fn calculate_object_reputation(object_reputation: &mut Vec<f64>,
    user_reputation: &Vec<f64>, ratings: &Vec<Rating>) {
    let mut object_weight_sum: Vec<f64> = vec![0.0; object_reputation.len()];

    for rep in object_reputation.iter_mut() {
        *rep = 0.0;
    }

    for r in ratings.iter() {
        object_reputation[r.object] += user_reputation[r.user] * r.weight;
        object_weight_sum[r.object] += user_reputation[r.user];
    }

    for (o, rep) in object_reputation.iter_mut().enumerate() {
        let w = object_weight_sum[o];
        if w > 0.0 { *rep /= w; }
    }
}

fn calculate_user_divergence(user_divergence: &mut Vec<f64>,
    object_reputation: &Vec<f64>, ratings: &Vec<Rating>) {
    for d in user_divergence.iter_mut() {
        *d = 0.0;
    }

    for r in ratings.iter() {
        let aux = r.weight - object_reputation[r.object];
        user_divergence[r.user] += aux * aux;
    }
}

fn calculate_user_reputation(user_reputation: &mut Vec<f64>,
    user_divergence: &Vec<f64>, user_links: &Vec<usize>,
    exponent: f64, min_divergence: f64) {
    for (u, rep) in user_reputation.iter_mut().enumerate() {
        if user_links[u] > 0 {
            let base = (user_divergence[u] / user_links[u] as f64) + min_divergence;
            *rep = base.powf(-exponent);
        } else {
            *rep = 0.0;
        }
    }
}


fn init_user_links(user_links: &mut Vec<usize>, ratings: &Vec<Rating>) {
    for l in user_links.iter_mut() {
        *l = 0;
    }

    for r in ratings.iter() {
        user_links[r.user] += 1;
    }
}


fn main() {
    use rand::{FromEntropy, Rng};
    use rand::distributions::{Distribution, Uniform};
    use rand::prng::{XorShiftRng};

    use std::f64;

    let objects: usize = 1000;
    let users: usize = 1000;

    let yzlm = YZLM {
        convergence: 1e-24,
        exponent: 0.8,
        min_divergence: 1e-36,
    };

    let mut object_quality: Vec<f64> = vec![f64::NAN; objects];
    let mut object_reputation: Vec<f64> = vec![f64::NAN; objects];

    let mut reputation_buf: Vec<f64> = vec![f64::NAN; objects];

    let mut user_divergence: Vec<f64> = vec![f64::NAN; users];
    let mut user_error: Vec<f64> = vec![f64::NAN; users];
    let mut user_links: Vec<usize> = vec![0; users];
    let mut user_reputation: Vec<f64> = vec![f64::NAN; users];

    let mut ratings: Vec<Rating> = Vec::new();

    let quality_dist = Uniform::new(0.0, 10.0);
    let error_dist = Uniform::new(0.0, 1.0);

    let mut rng = XorShiftRng::from_entropy();

    let mut iter_total = 0;

    for i in 0 .. 100 {
        for q in object_quality.iter_mut() {
            *q = quality_dist.sample(&mut rng);
        }

        for e in user_error.iter_mut() {
            *e = error_dist.sample(&mut rng);
        }

        for rep in user_reputation.iter_mut() {
            *rep = 1.0;
        }

        ratings.clear();
        for (object, q) in object_quality.iter().enumerate() {
            for (user, e) in user_error.iter().enumerate() {
                let weight = rng.gen_range(q - e, q + e);
                let r = Rating { object, user, weight };
                ratings.push(r);
            }
        }

        println!("[{}] Generated {} ratings", i, ratings.len());

        init_user_links(&mut user_links, &ratings);

        let mut iterations: usize = 0;
        let mut diff: f64 = f64::NAN;

        yzlm.calculate_reputation(&mut iterations, &mut diff,
            &mut object_reputation,
            &mut user_reputation, &user_links,
            &mut user_divergence, &mut reputation_buf,
            &ratings);

        println!("Exited in {} iterations with diff = {:e}",
                 iterations, diff);

        iter_total += iterations;

        let mut delta = 0.0;

        for (o, rep) in object_reputation.iter().enumerate() {
            delta += (rep - object_quality[o]).powf(2.0);
        }

        delta = (delta / object_quality.len() as f64).sqrt();

        println!("Error in quality estimate: {}", delta);
        println!("--------");
    }

    println!("Total iterations: {}", iter_total);
}
