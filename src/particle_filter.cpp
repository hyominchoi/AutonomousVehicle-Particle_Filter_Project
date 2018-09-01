/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <array>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    particles = {};
    weights = {};
    
    num_particles = 20;
    default_random_engine gen;
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    // cout << x << y << theta << particles.size() << endl;
    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle cur_particle;
        cur_particle.id = i;
        cur_particle.x = dist_x(gen);
        cur_particle.y = dist_y(gen);
        cur_particle.theta = dist_theta(gen);
        cur_particle.weight = 1.;
        cur_particle.associations = {};
        cur_particle.sense_x = {};
        cur_particle.sense_y = {};
        // append
        particles.push_back(cur_particle);
        weights.push_back(cur_particle.weight);
    }
    is_initialized = true;
    return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    
    for (unsigned int i = 0; i < particles.size(); ++i) {
        Particle cur_particle = particles[i];
        double x_f, y_f, theta_f;
        theta_f = cur_particle.theta + yaw_rate * delta_t;
        if (yaw_rate > 0.001) {
            x_f = cur_particle.x + velocity * (sin(theta_f) - sin(cur_particle.theta))/yaw_rate;
            y_f = cur_particle.y + velocity * (cos(cur_particle.theta) - cos(theta_f))/yaw_rate;
        }
        else {
            x_f = cur_particle.x + velocity * delta_t * cos(theta_f);
            y_f = cur_particle.y + velocity * delta_t * sin(theta_f);
        }
        normal_distribution<double> dist_x(x_f, std_x);
        normal_distribution<double> dist_y(y_f, std_y);
        normal_distribution<double> dist_theta(theta_f, std_theta);
        // update particle position
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
    
    return;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for (int i = 0; i < observations.size(); ++i) {
        double min_distance = numeric_limits<double>::max();
        int arg_min = -1;
        for (int j = 0; j <predicted.size(); ++j) {
            double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            if (distance < min_distance) {
                min_distance = distance;
                arg_min = j;
            }
        }
        observations[i].id = predicted[arg_min].id;
    }
    return;
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double gauss_norm = (1/(2*M_PI * sig_x * sig_y));
    
    // cout << "total observations " << observations.size() << endl;
    vector<LandmarkObs> predicted;
    for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {
        LandmarkObs obj;
        obj.id = map_landmarks.landmark_list[i].id_i;
        //note: map_landmarks id starts with 1, not 0
        obj.x = map_landmarks.landmark_list[i].x_f;
        obj.y = map_landmarks.landmark_list[i].y_f;
        predicted.push_back(obj);
    }
    
    for (int i = 0; i < particles.size(); ++i) {
        // for each particle, convert observations to world coordinates
        Particle cur_particle = particles[i];
        double x = cur_particle.x;
        double y = cur_particle.y;
        double theta = cur_particle.theta;
        vector<LandmarkObs> trans_obs;
        
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs cur_trans_obs;
            cur_trans_obs.id = observations[j].id;
            cur_trans_obs.x = x + cos(theta)*observations[j].x - sin(theta)*observations[j].y;
            cur_trans_obs.y = y + sin(theta)*observations[j].x + cos(theta)*observations[j].y;
            trans_obs.push_back(cur_trans_obs);
        }
        //cout << "transformation done for a particle i = " << i << endl;
        // find landmark data association using nearest neighborhood methd
        dataAssociation(predicted, trans_obs);
        vector<int> associations;
       
        vector<double> sense_x;
       
        vector<double> sense_y;
       
        //update weight using multivariate gaussian
        double weight = 1;
        for (int k = 0; k < trans_obs.size(); ++k) {
            int landmark_index = trans_obs[k].id - 1;
            //note: map_landmarks id starts with 1, not 0
            double mu_x = predicted[landmark_index].x;
            double mu_y = predicted[landmark_index].y;
            double exponent = ((trans_obs[k].x - mu_x)*(trans_obs[k].x - mu_x))/(2*sig_x*sig_x) + ((trans_obs[k].y - mu_y)*(trans_obs[k].y - mu_y))/(2*sig_y*sig_y);
            weight *= gauss_norm * exp(-exponent);
            
            sense_x.push_back(trans_obs[k].x);
            sense_y.push_back(trans_obs[k].y);
            associations.push_back(trans_obs[k].id);
        }
        SetAssociations(particles[i], associations, sense_x, sense_y);
        particles[i].weight = weight;
        weights[i] = (weight);
        
    }
    // int t;
    // cin >> t;
    // cout << "Particle Filter" << endl;
    return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    /*double total_weights = 0;
    for (int i = 0; i < weights.size(); ++i) {
        total_weights += weights[i];
    }
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = weights[i]/total_weights;
    }
    */
    vector<Particle> new_particles;
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> d(weights.begin(), weights.end());
    
    for (int i = 0; i < num_particles; ++i) {
        int index = d(gen);
        new_particles.push_back(particles[index]);
    }
    
    particles = new_particles;
    vector<Particle>().swap(new_particles);
    
    return;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
