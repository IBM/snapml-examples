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

#include "Dataset.h"
#include <sstream>
#include <fstream>
#include <cassert>

Dataset::Dataset(const std::string& filename) { load(filename); }

void Dataset::validate() const 
{
    if (num_ex <= 0)
        throw std::runtime_error("Invalid number of examples");

    if (num_ft <= 0)
        throw std::runtime_error("Invalid number of features");

    if (labels.size() != num_ex)
        throw std::runtime_error("Inconsistent number of labels");

    if (data.size() != num_ex * num_ft)
        throw std::runtime_error("Inconsistent number of examples");

    std::cout << "  Data OK. " << std::endl;
}

float Dataset::get_feature(uint32_t row, uint32_t col) const
{
    if (data.size() / num_ft <= row)
        throw std::runtime_error("Invalid row index");

    if (data.size() / num_ex <= col)
        throw std::runtime_error("Invalid column index");

    return data[row * num_ft + col];
}

float Dataset::get_label(uint32_t row) const
{
    if (labels.size() <= row)
        throw std::runtime_error("Invalid row index");

    return labels[row];
}

uint32_t Dataset::get_num_ex() const { return num_ex; }

uint32_t Dataset::get_num_ft() const { return num_ft; }

float* Dataset::get_data() { return data.data(); }

float* Dataset::get_labels() { return labels.data(); }

void Dataset::parse_line(const std::string& line)
{
    std::string       word;
    std::stringstream str(line);

    getline(str, word, ',');
    labels.push_back(std::stof(word));

    if (std::stof(word) < 0) {
        num_neg++;
    } else {
        num_pos++;
    }

    uint32_t n_ft = 0;
    while (getline(str, word, ',')) {
        data.push_back(std::stof(word));
        n_ft++;
    }

    num_ex++;

    if (num_ft == 0)
        num_ft = n_ft;
    else if (n_ft != num_ft) {
        throw std::runtime_error("Inconsistent number of features");
    }
}

void Dataset::load(const std::string& filename)
{
    std::fstream file = std::fstream(filename, std::ios::in);
    if (!file.is_open())
        throw std::runtime_error("Could not open file.");

    std::string line;
    while (getline(file, line)) {
        parse_line(line);
    }
}
