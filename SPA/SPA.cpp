#pragma warning(disable : 4996)
#include "IMGReader.h"
#include "Layers.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <random>
using namespace std;
namespace fs = std::filesystem;
vector<vector<vector<Num>>> inp;


std::string dirPath = "D:/Programming/C++Console/SPA/SPA/dataset/Reduced MNIST Data/Reduced Training data/"; // Replace with your image path
std::string testPath = "D:/Programming/C++Console/SPA/SPA/dataset/Reduced MNIST Data/Reduced Testing data/";


vector<vector<Num>> target =
{
    {1,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,1},
};

Convolutional2D conv1(5, 4, 4);
Convolutional2D conv2(3, 2, 0);
Linear l1(32, 20);
Linear l2(20, 10);

void Save()
{
    ofstream myfile;
    myfile.open("weights1.txt");
    
    myfile << conv1.filterSize << " " << conv1.filterNum << "\n";
    for (int t = 0; t < conv1.filterNum; t++)
    {
        for (int i = 0; i < conv1.filterSize; i++)
        {
            for (int j = 0; j < conv1.filterSize; j++)
            {
                myfile << conv1.filters[t][i][j] << " ";
            }
            myfile << "\n";
        }
    }


    myfile << conv2.filterSize << " " << conv2.filterNum << "\n";
    for (int t = 0; t < conv2.filterNum; t++)
    {
        for (int i = 0; i < conv2.filterSize; i++)
        {
            for (int j = 0; j < conv2.filterSize; j++)
            {
                myfile << conv2.filters[t][i][j] << " ";
            }
            myfile << "\n";
        }
    }

    myfile << l1.inpSize << " " << l1.outSize << "\n";
    for (int i = 0; i < l1.outSize; i++)
    {
        for (int j = 0; j < l1.inpSize; j++)
        {
            myfile << l1.weights[i][j] << " ";
        }
        myfile << "\n";
    }
    for (int i = 0; i < l1.outSize; i++)
    {
        myfile << l1.bias[i] << " ";
    }
    myfile << "\n";
    myfile << l2.inpSize << " " << l2.outSize << "\n";
    
    for (int i = 0; i < l2.outSize; i++)
    {
        for (int j = 0; j < l2.inpSize; j++)
        {
            myfile << l2.weights[i][j] << " ";
        }
        myfile << "\n";
    }
    for (int i = 0; i < l2.outSize; i++)
    {
        myfile << l2.bias[i] << " ";
    }
    

    myfile.close();
    cout << "Saved\n";
}


void Load()
{
    //freopen("weight1.txt", "r", stdin);
    ifstream fin;
    fin.open("weights1.txt");
    fin >> conv1.filterSize >> conv1.filterNum;
    for (int t = 0; t < conv1.filterNum; t++)
    {
        for (int i = 0; i < conv1.filterSize; i++)
        {
            for (int j = 0; j < conv1.filterSize; j++)
            {
                fin >> conv1.filters[t][i][j].val;
            }
        }
    }

    fin >> conv2.filterSize >> conv2.filterNum;
    for (int t = 0; t < conv2.filterNum; t++)
    {
        for (int i = 0; i < conv2.filterSize; i++)
        {
            for (int j = 0; j < conv2.filterSize; j++)
            {
                fin >> conv2.filters[t][i][j].val;
            }
        }
    }


    fin>>l1.inpSize >>l1.outSize;
    for (int i = 0; i < l1.outSize; i++)
    {
        for (int j = 0; j < l1.inpSize; j++)
        {
            fin >> l1.weights[i][j].val;
        }
    }
    for (int i = 0; i < l1.outSize; i++)
    {
        fin >> l1.bias[i].val;
    }

    fin >> l2.inpSize >> l2.outSize;
    for (int i = 0; i < l2.outSize; i++)
    {
        for (int j = 0; j < l2.inpSize; j++)
        {
            fin >> l2.weights[i][j].val;
        }
    }
    for (int i = 0; i < l2.outSize; i++)
    {
        fin >> l2.bias[i].val;
    }
    cout << "Loaded\n";
    fin.close();
    //Save();
}


double lr = 0.00005;

void Train(int epoch)
{
    cout << "Epoch " << epoch <<": " << "\n";
    bool resetGrad = true;
    double totalLoss = 0;
    for (int i = 0; i < 10; i++)
    {
        string dirImg = dirPath + to_string(i) + "/";
        int j = 0;
        for (const auto& entry : fs::directory_iterator(dirImg)) 
        {
            if (AddtionalFunctions::RanRange(0, 1) > 0.75) continue;
            j++;
            if (fs::is_regular_file(entry.status())) {
                string filename = entry.path().filename().string();
                convertImageToDoubleMatrix(dirImg + filename, 28, 28,inp);
                vector<vector<vector<Num>>> out;
                out = conv1.forward(inp);
                out = conv2.forward(out);

                vector<Num> out2 = Flatten(out);
                out2 = l1.forward(out2);
                out2 = l2.forward(out2);


             
                for (int t = 0; t < 10; t++)
                {
                    Num loss = MathFunction::Square(target[i][t] - out2[t]);
                    totalLoss += loss.val;
                    /*if (resetGrad)
                    {
                        AutoGradientSystem::ClearGrad(loss.curNode);
                    }*/
                    AutoGradientSystem::CalcGrad(loss.curNode, 0.5);
                    resetGrad = 0;
                }
            }
        }
        AutoGradientSystem::ClearEverything();

    }
    cout <<totalLoss << "\n";
    
    conv1.fit(lr);
    conv2.fit(lr);
    l1.fit(lr);
    l2.fit(lr);
}

void Train()
{
    for (int epoch = 0; epoch <= 100; epoch++)
    {
        Train(epoch);
        AutoGradientSystem::ClearEverything();
        if (epoch % 5 == 0)
        {
            Save();
        }
    }

}

int main() {
    //load weights
    Load();

    //prepare memory for image
    inp.assign(1, vector<vector<Num>>());
    inp[0].assign(28, vector<Num>());
    for (int i = 0; i < 28; i++)
    {
        inp[0][i].assign(28, Num());
    }

    //read image
    convertImageToDoubleMatrix("Test.jpg", 28, 28,inp);
    

    //print the image
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            cout << (inp[0][i][j].val > 0.5 ? 1 : 0);
        }
        cout << "\n";
    }


    //process
    vector<vector<vector<Num>>> out;
    out = conv1.forward(inp);
    out = conv2.forward(out);

    vector<Num> out2 = Flatten(out);
    out2 = l1.forward(out2);
    out2 = l2.forward(out2);



    //print result
    for (int i = 0; i < 10; i++)
    {
        cout << "Probability of number " << i << ": " << out2[i] << "\n";
    }

    cout << "You are drawing number: " << MathFunction::MaxIndex(out2) << "\n";

    return 0;
}
