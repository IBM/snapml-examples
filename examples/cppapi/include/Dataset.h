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

#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <iostream>

class Dataset {

public:
    Dataset(const std::string& filename);
    void     validate() const;
    float    get_feature(uint32_t row, uint32_t feature) const;
    float    get_label(uint32_t row) const;
    uint32_t get_num_ex() const;
    uint32_t get_num_ft() const;
    float*   get_data();
    float*   get_labels();

private:
    std::vector<float> data;
    std::vector<float> labels;

    uint32_t num_ex  = 0;
    uint32_t num_ft  = 0;
    uint32_t num_pos = 0;
    uint32_t num_neg = 0;

    void parse_line(const std::string& line);
    void load(const std::string& filename);
};

#endif
