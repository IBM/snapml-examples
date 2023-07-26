//# *****************************************************************
//#
//# Licensed Materials - Property of IBM
//#
//# (C) Copyright IBM Corp. 2023. All Rights Reserved.
//#
//# US Government Users Restricted Rights - Use, duplication or
//# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
//#
//# *****************************************************************

#include "TreeTypes.hpp"
#include "DenseDataset.hpp"
#include "DecisionTreeModel.hpp"
#include "DecisionTreePredictor.hpp"
#include "DecisionTreeBuilder.hpp"
#include "Dataset.h"
#include "DecisionTreeParams.hpp"
#include <cassert>

int main()
{
    std::cout << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Read input data and convert to Snap ML data format" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "  Loading train dataset" << std::endl;
    std::string filename = "../data/train_data.csv";
    Dataset     train_data(filename);
    train_data.validate();

    std::cout << "  Loading test dataset" << std::endl;
    filename = "../data/test_data.csv";
    Dataset test_data(filename);
    test_data.validate();

    snapml::DenseDataset snapml_train_data = snapml::DenseDataset(train_data.get_num_ex(), train_data.get_num_ft(),
                                                                  train_data.get_data(), train_data.get_labels());

    snapml::DenseDataset snapml_test_data = snapml::DenseDataset(test_data.get_num_ex(), test_data.get_num_ft(),
                                                                 test_data.get_data(), test_data.get_labels());

    std::cout << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Scenario 1 (training + inference): Decision Tree Model" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "  Training model " << std::endl;
    snapml::DecisionTreeParams  params;
    snapml::DecisionTreeBuilder builder = snapml::DecisionTreeBuilder(snapml_train_data, &params);
    builder.init();
    builder.build(nullptr);

    std::cout << "  Running inference  " << std::endl;
    snapml::DecisionTreeModel     model     = builder.get_model();
    snapml::DecisionTreePredictor predictor = snapml::DecisionTreePredictor(model);

    std::vector<double> preds(test_data.get_num_ex());
    uint32_t            inference_threads = 4;
    predictor.predict(snapml_test_data, preds.data(), inference_threads);

    for (uint32_t i = 0; i < preds.size(); i++) {
        if (preds[i] < 0)
            preds[i] = 0;
    }

    uint32_t acc_preds = 0;
    for (uint32_t i = 0; i < preds.size(); i++) {
        acc_preds += (preds[i] == test_data.get_label(i));
    }

    double acc = acc_preds * 1.0 / test_data.get_num_ex();
    std::cout << "  Accuracy score " << acc << std::endl;
    assert(acc == 1.0);

    std::cout << std::endl;

    return 0;
}
