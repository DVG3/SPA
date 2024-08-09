#pragma once
#include <omp.h>
#include <vector>
//#include "SPALib.h"
using namespace std;
struct Linear
{
    int outSize, inpSize;
    vector<vector<Num>> weights;
    vector<Num> bias;
    Num(*activateFunc)(Num) = MathFunction::Sigmoid;
    Linear(int inpSize, int outSize) :outSize(outSize), inpSize(inpSize)
    {
        weights.assign(outSize, vector<Num>());
        bias.assign(outSize, Num());

        for (int i = 0; i < outSize; i++)
        {
            weights[i].assign(inpSize, Num());
            for (int j = 0; j < inpSize; j++)
            {
                weights[i][j] = AddtionalFunctions::RanRange(-1, 1);
            }
            bias[i] = AddtionalFunctions::RanRange(-1, 1);
        }
    }


    vector<Num> forward(const vector<Num>& inp)
    {
        vector<Num> out;
        out.assign(outSize, Num());
        for (int i = 0; i < outSize; i++)
        {
            out[i] = 0;
            for (int j = 0; j < inpSize; j++)
            {
                out[i] = out[i] + weights[i][j] * inp[j];
            }
            out[i] = out[i] + bias[i];
            out[i] = activateFunc(out[i]);
        }
        return out;
    }

    void fit(double lr)
    {
        #pragma omp parallel for
        for (int i = 0; i < outSize; i++)
        {
            for (int j = 0; j < inpSize; j++)
            {
                weights[i][j].apply(lr);
                weights[i][j].curNode->grad = 0;
            }
            bias[i].apply(lr);
            bias[i].curNode->grad = 0;
        }
    }

};

struct Convolutional2D
{
    int filterSize = 0, filterNum = 0, padding = 0;
    vector<vector<vector<Num>>> filters;
    Convolutional2D(int filterSize, int filterNum, int padding = 0):filterNum(filterNum),filterSize(filterSize),padding(padding)
    {
        filters.assign(filterNum, vector<vector<Num>>());
        for (int t = 0; t < filterNum; t++)
        {
            filters[t].assign(filterSize, vector<Num>());

            for (int i = 0; i < filterSize; i++)
            {
                filters[t][i].assign(filterSize, Num());
                for (int j = 0; j < filterSize; j++)
                {
                    filters[t][i][j] = AddtionalFunctions::RanRange(-1, 1);
                }
            }
        }
    }

    vector<vector<vector<Num>>> forward(const vector<vector<vector<Num>>>& inp)
    {

        vector<vector<vector<Num>>> out;
        out.assign(inp.size() * filterNum, vector<vector<Num>>());
        int idx = 0;
        for (int t2 = 0; t2 < filterNum; t2++)
        {
            for (int t1 = 0; t1 < inp.size(); t1++)
            {
                int h = 0;
                out[idx].assign((inp[t1].size() - filterSize - 1) / (padding + 1) + 1, vector<Num>());

                for (int i = 0; i < inp[t1].size() - filterSize; i+= padding + 1)
                {
                    int w = 0;
                    out[idx][h].assign((inp[t1][i].size() - filterSize - 1) / (padding + 1) + 1, Num());
                    for (int j = 0; j < inp[t1][i].size() - filterSize; j+= padding + 1)
                    {
                        for (int fh = 0; fh < filterSize; fh++)
                        {
                            for (int fw = 0; fw < filterSize; fw++)
                            {
                                out[idx][h][w] = out[idx][h][w] +  filters[t2][fh][fw] * inp[t1][i + fh][j + fw];
                             }
                        }
                        w++;
                    }
                    h++;
                }
                idx++;
            }

        }
        return out;
    }

    void fit(double lr)
    {
#pragma omp parallel for
        for (int t = 0; t < filters.size(); t++)
        {
            for (int i = 0; i < filterSize; i++)
            {
                for (int j = 0; j < filterSize; j++)
                {
                    filters[t][i][j].apply(lr);
                    filters[t][i][j].curNode->grad = 0;
                }
            }
        }
    }
};

vector<Num> Flatten(const vector<vector<vector<Num>>>& inp)
{
    vector<Num> out;
    out.assign(inp.size() * inp[0].size() * inp[0][0].size(), Num());
    int idx = 0;
    for (int t = 0; t < inp.size(); t++)
    {
        for (int i = 0; i < inp[t].size(); i++)
        {
            for (int j = 0; j < inp[t][i].size(); j++)
            {
                out[idx] = inp[t][i][j];
                
                idx++;
            }
        }
    }
    return out;
}