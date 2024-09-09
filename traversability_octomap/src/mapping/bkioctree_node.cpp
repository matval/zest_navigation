#include <algorithm>
#include <assert.h>

#include "bkioctree_node.h"

namespace traversability_bki
{
    // Default parameters
    float Traversability::sf2 = 1.0f;
    float Traversability::ell = 1.0f;
    float Traversability::mean_prior = 0.5f;
    float Traversability::lambda_prior = 4.0f;
    float Traversability::beta_prior = 1.0f;
    float Traversability::prior_A = 1.0f;
    float Traversability::prior_B = 1.0f;
    float Traversability::var_thresh = 1000.0f;
    float Traversability::free_thresh = 0.3f;
    float Traversability::occupied_thresh = 0.7f;

    float Traversability::get_occupancy_prob()
    {
        return this->m_A / (this->m_A + this->m_B);
    }

    float Traversability::get_occupancy_var()
    {
        float num = this->m_A * this->m_B;
        float den = (this->m_A + this->m_B) * (this->m_A + this->m_B) * (this->m_A + this->m_B + 1.0f);

        return num / den;
    }

    void Traversability::update(std::vector<float>& y)
    {
        this->measured = true;

        // Update the Normal-Inverse-Gamma distribution
        this->beta += 0.5f * this->lambda * y[1] * (y[0] - this->mean) * (y[0] - this->mean) / (this->lambda + y[1]);
        this->mean = (this->lambda * this->mean + y[1] * y[0]) / (this->lambda + y[1]);
        this->lambda += y[1];
        this->alpha += 0.5f * y[1];

        // Update the Bernoulli distribution
        this->m_A += y[1];
        this->m_B += y[2] - y[1];

        float var = this->get_occupancy_var();
        if (var > Traversability::var_thresh) {
            this->state = State::UNKNOWN;
        }
        else {
            float prob = get_occupancy_prob();
            if (prob > Traversability::occupied_thresh) {
                this->state = State::OCCUPIED;
            }
            else if (prob < Traversability::free_thresh) {
                this->state = State::FREE;
            }
            else {
                this->state = State::UNKNOWN;
            }
        }
    }
}
