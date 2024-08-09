#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "SPALib.h"
#include <random>
using namespace std;

double lr = 0.01;
double lrDecay = 0.99999;
Num w1 = RanRange(-1, 1);
Num w2 = RanRange(-1, 1);
Num bias = RanRange(-1, 1);
const int testPoint = 10;
pair<Num, Num> points[testPoint];

double evalFunc(const pair<double, double>& point)
{
	return w1.val * point.first + w2.val * point.second + bias.val;
}

void train(int epoch)
{
	lr *= pow(lrDecay, epoch);
	double totalLoss = 0;

	Num loss = 0;
	for (int i = 0; i < testPoint; i++)
	{
		{
			Num pred = points[i].first * w1 + points[i].second * w2 + bias;

			Num target = 0;

			loss = (target - pred) * (target - pred);
			totalLoss += loss.val;


			//cerr << "Deb: " << w1.grad << " " << w2.grad << " " << bias.grad << "\n";

			AutoGradientSystem::ClearGrad(loss.curNode);
			AutoGradientSystem::CalcGrad(loss.curNode);

		}
	}
	w1.val = w1.val - loss.val * lr / w1.grad();
	w2.val = w2.val - loss.val * lr / w2.grad();
	bias.val = bias.val - loss.val * lr / bias.grad();
	if (epoch % 10 == 0)
	{
		cout << "epoch " << epoch << ": ";

		cout << totalLoss << "\n";

	}


}

int main()
{
	//w1.curNode->beDed = 0;
	//w2.curNode->beDed = 0;
	//bias.curNode->beDed = 0;
	for (int i = 0; i < testPoint; i++)
	{
		points[i] = { RanRange(0,5), RanRange(0,5) };
		//points[i].first.curNode->beDed = 0;
		//points[i].second.curNode->beDed = 0;

	}
	double totalLoss = 0;
	for (int i = 0; i < testPoint; i++)
	{
		totalLoss += evalFunc({ points[i].first.val, points[i].second.val }) * evalFunc({ points[i].first.val, points[i].second.val });

	}

	cout << "Before train: " << totalLoss << "\n";


	for (int epoch = 1; epoch <= 10000; epoch++)
	{

		train(epoch);
		AutoGradientSystem::ClearEverything();

	}
}