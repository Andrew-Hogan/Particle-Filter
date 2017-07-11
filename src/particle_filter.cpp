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

#include "particle_filter.h"

#define NOZERO 0.0001
#define MAXTHRESH 1000000

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Gaussian Distribution
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	default_random_engine gen;
	num_particles = 50;

	for (int i = 0; i < num_particles; ++i) {
		Particle p = { i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0 };
		particles.push_back(p);
	}

	weights.resize(num_particles);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;

	// per particle
	for (int i = 0; i < num_particles; ++i) {
		Particle &particle = particles[i];

		if (fabs(yaw_rate) > NOZERO) {
			particle.x = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			particle.theta = particle.theta + delta_t * yaw_rate;

		}
		else {
			particle.x = particle.x + velocity * delta_t * cos(particle.theta);
			particle.y = particle.y + velocity * delta_t * sin(particle.theta);
		}

		normal_distribution<double> dist_x(particle.x, std_pos[0]);
		normal_distribution<double> dist_y(particle.y, std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// per observation
	for (int o = 0; o < observations.size(); o++) {
		double thresh = MAXTHRESH;
		// per prediction
		for (int p = 0; p < predicted.size(); p++) {
			LandmarkObs obs = observations[o];
			LandmarkObs pred = predicted[p];
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			
			// update best guess
			if (distance < thresh) {
				thresh = distance;
				observations[o].id = p;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// per particle
	for (int i = 0; i < num_particles; ++i) {
		Particle p = particles[i];
		double w = 1.0;

		// per observation
		for (int o = 0; o < observations.size(); o++) {
			Map::single_landmark_s nearest_lm = { 0, 0.0, 0.0 };
			double obs_x = p.x + observations[o].x * cos(p.theta) - observations[o].y * sin(p.theta);
			double obs_y = p.y + observations[o].x * sin(p.theta) + observations[o].y * cos(p.theta);
			double min_dist_lm = sensor_range;

			// per landmark
			for (int lm = 0; lm < map_landmarks.landmark_list.size(); lm++) {
				double p_lm = dist(p.x, p.y, map_landmarks.landmark_list[lm].x_f, map_landmarks.landmark_list[lm].y_f);
				if (p_lm <= sensor_range) {
					double o_lm = dist(obs_x, obs_y, map_landmarks.landmark_list[lm].x_f, map_landmarks.landmark_list[lm].y_f);
					if (o_lm < min_dist_lm) {
						min_dist_lm = o_lm;
						nearest_lm = map_landmarks.landmark_list[lm];
					}
				}
			}

			double lm_x = std_landmark[0];
			double lm_y = std_landmark[1];
			double delta_x = nearest_lm.x_f - obs_x;
			double delta_y = nearest_lm.y_f - obs_y;

			// Multivariate Gaussian Weight
			double obs_w = exp(-0.5 * ((pow(delta_x, 2) / pow(lm_x, 2)) + (pow(delta_y, 2) / pow(lm_y, 2)))) / (2 * M_PI * lm_x * lm_y);

			w = w * obs_w;
		}

		particles[i].weight = w;
		weights[i] = w;
	}

	double w_accumulated = accumulate(weights.begin(), weights.end(), 0.0);
	// update weights per particle
	for (int i = 0; i < num_particles; ++i) {
		particles[i].weight = particles[i].weight / w_accumulated;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	vector<Particle> new_ps;
	default_random_engine gen;

	discrete_distribution<> dist(weights.begin(), weights.end());

	// resample per particle
	for (int i = 0; i < num_particles; ++i) {
		int loc = dist(gen);
		Particle new_p = particles[loc];
		new_ps.push_back(new_p);
	}

	particles = new_ps;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
