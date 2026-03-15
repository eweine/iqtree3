#include "modelpetracer.h"

#include <cstring>
#include <sstream>

ModelPETracer::ModelPETracer(const char *model_name, string model_params, StateFreqType freq, string freq_params, PhyloTree *tree)
    : ModelMarkov(tree, false) {
    init(model_name, model_params, freq, freq_params);
}

bool ModelPETracer::validModelName(const string &model_name) {
    return model_name == "PETRACER" || model_name == "PE";
}

void ModelPETracer::initializeRootPrior() {
    state_freq[0] = 1.0;
    for (int state = 1; state < num_states; ++state) {
        state_freq[state] = 0.0;
    }
    freq_type = FREQ_USER_DEFINED;
    highest_freq_state = 0;
}

void ModelPETracer::init(const char *model_name, string model_params, StateFreqType freq, string freq_params) {
    name = validModelName(model_name) ? "PETRACER" : model_name;
    full_name = "PETracer irreversible edit model";
    initializeRootPrior();

    if (num_states < 2) {
        outError("PETRACER model requires at least 2 states");
    }

    if (!freq_params.empty() || (freq != FREQ_UNKNOWN && freq != FREQ_EQUAL && freq != FREQ_USER_DEFINED)) {
        outWarning("PETRACER ignores state-frequency modifiers and uses a fixed root prior on state 0.");
    }

    for (int i = 0; i < getNumRateEntries(); ++i) {
        rates[i] = 1.0;
    }
    num_params = getNumRateEntries();

    if (!model_params.empty()) {
        readRates(model_params);
        fixed_parameters = !Params::getInstance().optimize_from_given_params;
    }

    decomposeRateMatrix();
}

void ModelPETracer::startCheckpoint() {
    checkpoint->startStruct("ModelPETracer");
}

void ModelPETracer::saveCheckpoint() {
    startCheckpoint();
    CKP_ARRAY_SAVE(getNumRateEntries(), rates);
    endCheckpoint();
    ModelMarkov::saveCheckpoint();
}

void ModelPETracer::restoreCheckpoint() {
    ModelMarkov::restoreCheckpoint();
    startCheckpoint();
    CKP_ARRAY_RESTORE(getNumRateEntries(), rates);
    endCheckpoint();
    decomposeRateMatrix();
    if (phylo_tree) {
        phylo_tree->clearAllPartialLH();
    }
}

int ModelPETracer::getNumRateEntries() {
    return num_states - 1;
}

int ModelPETracer::getNDim() {
    return fixed_parameters ? 0 : num_params;
}

string ModelPETracer::getName() {
    return name;
}

string ModelPETracer::getNameParams(bool show_fixed_params) {
    ostringstream out;
    out << name;
    if (!fixed_parameters || show_fixed_params) {
        out << '{';
        for (int i = 0; i < getNumRateEntries(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << rates[i];
        }
        out << '}';
    }
    return out.str();
}

void ModelPETracer::writeInfo(ostream &out) {
    out << "State 0 is interpreted as the unedited root state" << endl;
    out << "Edit rates:";
    for (int state = 1; state < num_states; ++state) {
        out << " 0->" << state << ": " << rates[state - 1];
    }
    out << endl;
    out << "Root prior: state 0 fixed to 1.0; edited states fixed to 0.0" << endl;
}

void ModelPETracer::writeParameters(ostream &out) {
    for (int i = 0; i < getNumRateEntries(); ++i) {
        out << "\t" << rates[i];
    }
}

void ModelPETracer::readRates(istream &in) noexcept(false) {
    for (int i = 0; i < getNumRateEntries(); ++i) {
        string tmp_value;
        in >> tmp_value;
        if (tmp_value.empty()) {
            throw name + string(": Edit rates could not be read");
        }
        rates[i] = convert_double_with_distribution(tmp_value.c_str(), true);
        if (rates[i] < 0.0) {
            throw "Negative edit rates found";
        }
    }
}

void ModelPETracer::readRates(string str) noexcept(false) {
    int end_pos = 0;
    char separator = (str.find('/') != std::string::npos) ? '/' : ',';
    for (int i = 0; i < getNumRateEntries(); ++i) {
        int new_end_pos = 0;
        rates[i] = convert_double_with_distribution(str.substr(end_pos).c_str(), new_end_pos, true, separator);
        end_pos += new_end_pos;
        if (rates[i] < 0.0) {
            outError("Negative edit rates found in ", str);
        }
        if (i == getNumRateEntries() - 1) {
            if (end_pos < str.length()) {
                outError("String too long ", str);
            }
        } else {
            if (end_pos >= str.length()) {
                outError("The number of input edit rates (" + convertIntToString(i + 1) + ") is less than the number of edited states (" + convertIntToString(getNumRateEntries()) + ").");
            }
            if (str[end_pos] != ',' && str[end_pos] != '/') {
                outError("Comma or forward slash to separate edit rates not found in ", str);
            }
            ++end_pos;
        }
    }
    num_params = getNumRateEntries();
}

void ModelPETracer::getRateMatrix(double *rate_mat) {
    memcpy(rate_mat, rates, sizeof(double) * getNumRateEntries());
}

void ModelPETracer::setRateMatrix(double *rate_mat) {
    memcpy(rates, rate_mat, sizeof(double) * getNumRateEntries());
    decomposeRateMatrix();
    if (phylo_tree) {
        phylo_tree->clearAllPartialLH();
    }
}

void ModelPETracer::getStateFrequency(double *freq, int mixture) {
    initializeRootPrior();
    memcpy(freq, state_freq, sizeof(double) * num_states);
}

void ModelPETracer::setStateFrequency(double *freq) {
    initializeRootPrior();
}

double ModelPETracer::getScaledEditRateSum() const {
    return -rate_matrix[0];
}

void ModelPETracer::computeTransMatrix(double time, double *trans_matrix, int mixture, int selected_row) {
    memset(trans_matrix, 0, sizeof(double) * num_states * num_states);

    double lambda = getScaledEditRateSum();
    double stay_unedited = exp(-lambda * time);
    double leave_unedited = 1.0 - stay_unedited;

    auto fill_row = [&](int row) {
        if (row == 0) {
            trans_matrix[0] = stay_unedited;
            if (lambda > 0.0) {
                for (int state = 1; state < num_states; ++state) {
                    trans_matrix[state] = leave_unedited * rate_matrix[state] / lambda;
                }
            }
        } else {
            trans_matrix[row * num_states + row] = 1.0;
        }
    };

    if (selected_row >= 0) {
        fill_row(selected_row);
        return;
    }

    fill_row(0);
    for (int row = 1; row < num_states; ++row) {
        fill_row(row);
    }
}

double ModelPETracer::computeTrans(double time, int state1, int state2) {
    double lambda = getScaledEditRateSum();
    if (state1 == 0) {
        if (state2 == 0) {
            return exp(-lambda * time);
        }
        return (lambda > 0.0) ? (1.0 - exp(-lambda * time)) * rate_matrix[state2] / lambda : 0.0;
    }
    return state1 == state2 ? 1.0 : 0.0;
}

double ModelPETracer::computeTrans(double time, int state1, int state2, double &derv1, double &derv2) {
    double lambda = getScaledEditRateSum();
    if (state1 == 0) {
        double stay_unedited = exp(-lambda * time);
        if (state2 == 0) {
            derv1 = -lambda * stay_unedited;
            derv2 = lambda * lambda * stay_unedited;
            return stay_unedited;
        }
        double rate = rate_matrix[state2];
        derv1 = rate * stay_unedited;
        derv2 = -rate * lambda * stay_unedited;
        return (lambda > 0.0) ? (1.0 - stay_unedited) * rate / lambda : 0.0;
    }
    derv1 = 0.0;
    derv2 = 0.0;
    return state1 == state2 ? 1.0 : 0.0;
}

void ModelPETracer::computeTransDerv(double time, double *trans_matrix, double *trans_derv1, double *trans_derv2, int mixture) {
    memset(trans_derv1, 0, sizeof(double) * num_states * num_states);
    memset(trans_derv2, 0, sizeof(double) * num_states * num_states);
    computeTransMatrix(time, trans_matrix, mixture);

    double lambda = getScaledEditRateSum();
    double stay_unedited = exp(-lambda * time);
    trans_derv1[0] = -lambda * stay_unedited;
    trans_derv2[0] = lambda * lambda * stay_unedited;
    for (int state = 1; state < num_states; ++state) {
        double rate = rate_matrix[state];
        trans_derv1[state] = rate * stay_unedited;
        trans_derv2[state] = -rate * lambda * stay_unedited;
    }
}

void ModelPETracer::decomposeRateMatrix() {
    initializeRootPrior();
    memset(rate_matrix, 0, sizeof(double) * num_states * num_states);

    double raw_sum = 0.0;
    for (int state = 1; state < num_states; ++state) {
        raw_sum += rates[state - 1];
    }
    if (raw_sum <= 0.0) {
        throw "PETRACER requires at least one positive edit rate";
    }

    double scale = total_num_subst / raw_sum;
    for (int state = 1; state < num_states; ++state) {
        rate_matrix[state] = rates[state - 1] * scale;
    }
    rate_matrix[0] = -total_num_subst;
}

void ModelPETracer::setBounds(double *lower_bound, double *upper_bound, bool *bound_check) {
    int ndim = getNDim();
    for (int i = 1; i <= ndim; ++i) {
        lower_bound[i] = 0.0;
        upper_bound[i] = MAX_RATE;
        bound_check[i] = false;
    }
}
