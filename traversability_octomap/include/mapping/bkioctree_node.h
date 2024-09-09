#pragma once

#include <vector>
#include <iostream>

namespace traversability_bki
{
    /// Occupancy state: before pruning: FREE, OCCUPIED, UNKNOWN; after pruning: PRUNED
    enum class State : char {
        FREE, OCCUPIED, UNKNOWN, PRUNED
    };

    /*
     * @brief Inference ouputs and occupancy state.
     *
     * Occupancy has member variables: m_A and m_B (kernel densities of positive
     * and negative class, respectively) and State.
     * Before using this class, set the static member variables first.
     */
    class Traversability
    {
        friend class TraversabilityBKIOctoMap;

    public:
        /*
         * @brief Constructors and destructor.
         */
        Traversability() : mean(mean_prior), lambda(lambda_prior), alpha(0.5*lambda_prior), beta(beta_prior), m_A(prior_A), m_B(prior_B), state(State::UNKNOWN) {
            this->measured = false;
        }

        Traversability(const Traversability &other) : state(other.state), mean(other.mean), lambda(other.lambda), alpha(other.alpha), beta(other.beta), m_A(other.m_A), m_B(other.m_B) { }

        Traversability &operator=(const Traversability &other)
        {
            this->state = other.state;
            this->mean = other.mean;
            this->lambda = other.lambda;
            this->alpha = other.alpha;
            this->beta = other.beta;
            this->m_A = other.m_A;
            this->m_B = other.m_B;
            return *this;
        }

        ~Traversability() { }

        /*
         * @brief Exact updates for nonparametric Bayesian kernel inference
         * @param ybar kernel density estimate of positive class (occupied)
         * @param kbar kernel density of negative class (unoccupied)
         */
        void update(std::vector<float>& y);

        // Get the probability of occupancy
        float get_occupancy_prob();

        // Get the variance of occupancy
        float get_occupancy_var();
        
        /*
         * @brief Get occupancy state of the node.
         * @return occupancy state (see State).
         */
        inline State get_state() const { return this->state; }

        // Get the mean of the node
        inline float get_mean() const { return this->mean; }

        // Get the lambda of the node
        inline float get_lambda() const { return this->lambda; }

        // Get the alpha of the node
        inline float get_alpha() const { return this->alpha; }

        // Get the beta of the node
        inline float get_beta() const { return this->beta; }
        
        // Prune current node; set state to PRUNED.
        inline void prune() { this->state = State::PRUNED; }

        // Only FREE and OCCUPIED nodes can be equal.
        inline bool operator==(const Traversability &rhs) const {
            return this->state != State::UNKNOWN && this->state == rhs.state;
        }

        bool measured;

    private:
        State state;

        // Parametrized as an Normal-Inverse-Gamma distribution
        double mean;
        double lambda;
        double alpha;
        double beta;
        double m_A;
        double m_B;
        
        static float sf2;
        static float ell;           // length-scale
        static float mean_prior;    // prior on mean
        static float lambda_prior;  // prior on precision
        static float beta_prior;    // prior on scale
        static float prior_A;       // prior on A
        static float prior_B;       // prior on B

        static float var_thresh;
        static float free_thresh;     // FREE occupancy threshold
        static float occupied_thresh; // OCCUPIED occupancy threshold
    };

    // Overloading the << operator for semantic_bki::State
    inline std::ostream& operator<<(std::ostream& os, const State& state)
    {
        switch (state)
        {
            case State::FREE:
                os << "FREE";
                break;
            case State::OCCUPIED:
                os << "OCCUPIED";
                break;
            case State::UNKNOWN:
                os << "UNKNOWN";
                break;
            case State::PRUNED:
                os << "PRUNED";
                break;
        }

        return os;  // Return the stream
    }

    typedef Traversability TraversabilityOcTreeNode;
}
