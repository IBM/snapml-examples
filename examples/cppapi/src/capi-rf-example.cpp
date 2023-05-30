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
#include "RandomForestModel.hpp"
#include "RandomForestPredictor.hpp"
#include "RandomForestBuilder.hpp"
#include "Dataset.h"
#include <cassert>

int main()
{
    std::cout << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Read input data and convert to Snap ML data format" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "  Loading the train dataset " << std::endl;
    std::string filename = "../data/train_data.csv";
    Dataset     train_data(filename);
    train_data.validate();

    std::cout << "  Loading the test dataset " << std::endl;
    filename = "../data/test_data.csv";
    Dataset test_data(filename);
    test_data.validate();

    snapml::DenseDataset snapml_train_data = snapml::DenseDataset(train_data.get_num_ex(), train_data.get_num_ft(),
                                                                  train_data.get_data(), train_data.get_labels());

    snapml::DenseDataset snapml_test_data = snapml::DenseDataset(test_data.get_num_ex(), test_data.get_num_ft(),
                                                                 test_data.get_data(), test_data.get_labels());

    std::cout << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Scenario 1 (training + inference): Random Forest Model" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "  Training model " << std::endl;
    snapml::RandomForestParams  params;
    snapml::RandomForestBuilder builder = snapml::RandomForestBuilder(snapml_train_data, &params);
    builder.init();
    builder.build(nullptr);

    std::cout << "  Running inference " << std::endl;
    snapml::RandomForestModel model = builder.get_model();

    snapml::DenseDataset dummy_data(0, 0, nullptr, nullptr);
    if (model.compressed_tree() == false) {
        model.compress(dummy_data); // Optimize model for CPU inference
        if (model.compressed_tree() == false) {
            std::cout << "  SnapML tree model optimization failed" << std::endl;
        }
    }
    snapml::RandomForestPredictor predictor = snapml::RandomForestPredictor(model);

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
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Scenario 2 (import + inference): Random Forest Model" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "  Importing model from file " << std::endl;
    snapml::RandomForestModel model_imp;
    model_imp.import_model("../models/model.pmml", "pmml", snapml::task_t::classification);

    if (model_imp.compressed_tree() == false) {
        model_imp.compress(dummy_data); // Optimize model for CPU inference
        if (model_imp.compressed_tree() == false) {
            std::cout << "  SnapML tree model optimization failed" << std::endl;
        }
    }

    uint32_t num_classes = model_imp.get_num_classes();
    std::cout << "  Number of classes found in the imported model " << num_classes << std::endl;

    std::cout << "  Running inference " << std::endl;
    snapml::RandomForestPredictor predictor_imp = snapml::RandomForestPredictor(model_imp);
    predictor_imp.predict(snapml_test_data, preds.data(), inference_threads);

    for (uint32_t i = 0; i < preds.size(); i++) {
        if (preds[i] < 0)
            preds[i] = 0;
        else
            preds[i] = 1;
    }

    acc_preds = 0;
    for (uint32_t i = 0; i < preds.size(); i++) {
        acc_preds += (preds[i] == test_data.get_label(i));
    }

    acc = acc_preds * 1.0 / test_data.get_num_ex();
    std::cout << "  Accuracy score " << acc << std::endl;
    assert(acc == 1.0);

    std::cout << std::endl;

    return 0;
}
