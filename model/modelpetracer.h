#ifndef MODELPETRACER_H_
#define MODELPETRACER_H_

#include "modelmarkov.h"

/*
 * Irreversible edit model for PETracer-style lineage characters.
 * State 0 is the unedited root state. Each edited state is absorbing.
 */
class ModelPETracer : public ModelMarkov {
public:
    ModelPETracer(const char *model_name, string model_params, StateFreqType freq, string freq_params, PhyloTree *tree);

    virtual void init(const char *model_name, string model_params, StateFreqType freq, string freq_params) override;
    virtual void startCheckpoint() override;
    virtual void saveCheckpoint() override;
    virtual void restoreCheckpoint() override;
    virtual int getNumRateEntries() override;
    virtual int getNDim() override;
    virtual string getName() override;
    virtual string getNameParams(bool show_fixed_params = false) override;
    virtual void writeInfo(ostream &out) override;
    virtual void writeParameters(ostream &out) override;
    virtual void readRates(istream &in) noexcept(false) override;
    virtual void readRates(string str) noexcept(false) override;
    virtual void getRateMatrix(double *rate_mat) override;
    virtual void setRateMatrix(double *rate_mat) override;
    virtual void getStateFrequency(double *freq, int mixture = 0) override;
    virtual void setStateFrequency(double *freq) override;
    virtual void computeTransMatrix(double time, double *trans_matrix, int mixture = 0, int selected_row = -1) override;
    virtual double computeTrans(double time, int state1, int state2) override;
    virtual double computeTrans(double time, int state1, int state2, double &derv1, double &derv2) override;
    virtual void computeTransDerv(double time, double *trans_matrix, double *trans_derv1, double *trans_derv2, int mixture = 0) override;
    virtual void decomposeRateMatrix() override;
    virtual void setBounds(double *lower_bound, double *upper_bound, bool *bound_check) override;

    static bool validModelName(const string &model_name);

protected:
    void initializeRootPrior();
    double getScaledEditRateSum() const;
};

#endif
