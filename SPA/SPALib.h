#pragma one
#include <random>
#include <iostream>
#include <vector>
#include <cmath>
typedef struct Node Node;
typedef struct Num Num;

namespace AddtionalFunctions
{
	double RanRange(double min, double max) {
		// Ensure that the random number generator is seeded only once
		static std::mt19937 gen(9191);

		// Create a uniform real distribution to generate doubles in the range [min, max]
		std::uniform_real_distribution<double> dis(min, max);

		// Return a random double
		return dis(gen);
	}
	double MapRange(double value, double fromMin, double fromMax, double toMin, double toMax) {
		// Ensure the original range is valid
		if (fromMin == fromMax) {
			throw std::invalid_argument("Original range cannot be zero length.");
		}

		// Calculate the proportion of the value in the original range
		double proportion = (value - fromMin) / (fromMax - fromMin);

		// Map the proportion to the target range
		double mappedValue = toMin + (proportion * (toMax - toMin));

		return mappedValue;
	}

	
}

struct Node
{
	bool beDed = 1;
	Node* first = 0; Node* second = 0;
	double grad = 0;
	double w1 = 0, w2 = 0;
	int operCount = 0;
	int ownerCount = 0;
	bool dontDestroyed = false;
};


std::vector<Node*> sortedNodes;

struct Num
{
	Node* curNode = 0;
	double adaptiveGradient = 0;
	double val = 0;

	void apply(double lr)
	{
		//normal
		//val -= lr * grad();
		//return;

		//adaptive
		adaptiveGradient += grad() * grad();
		val -= (lr/(std::sqrt(adaptiveGradient + 0.00000001)) )*grad();
	}

	void init(bool isTemp)
	{
		curNode = new Node();
		curNode->ownerCount++;
		sortedNodes.push_back(curNode);

		curNode->beDed = isTemp;
	}
	double grad()
	{
		return curNode->grad;
	}

	Num(double val) :val(val) { init(false); }
	Num() { init(true); }


	~Num()
	{
		curNode->ownerCount--;
	}


	Num(const Num& other)
	{
		this->curNode = other.curNode;
		this->val = other.val;
		this->curNode->ownerCount++;
	}

	Num& operator=(const Num& other) {
		if (this != &other) { // Self-assignment check
			if (curNode) curNode->ownerCount--;
			other.curNode->ownerCount++;
			this->curNode = other.curNode;
			this->val = other.val;

			//this->curNode->ownerCount++;
		}
		return *this;
	}
	Num operator+(Num num)
	{
		//num.destroyNodeAfterDeallocated = false;
		Num ret = 0;
		ret.val = val + num.val;
		ret.curNode->first = this->curNode;
		ret.curNode->second = num.curNode;
		ret.curNode->w1 = 1;
		ret.curNode->w2 = 1;

		this->curNode->operCount++;
		num.curNode->operCount++;

		return ret;
	}

	Num operator-(Num num)
	{
		//num.destroyNodeAfterDeallocated = false;

		Num ret = 0;
		ret.val = val - num.val;
		ret.curNode->first = this->curNode;
		ret.curNode->second = num.curNode;
		ret.curNode->w1 = 1;
		ret.curNode->w2 = -1;
		this->curNode->operCount++;
		num.curNode->operCount++;

		return ret;
	}

	Num operator*(Num num)
	{
		//num.destroyNodeAfterDeallocated = false;

		Num ret = 0;
		ret.val = val * num.val;
		ret.curNode->first = this->curNode;
		ret.curNode->second = num.curNode;
		ret.curNode->w1 = num.val;
		ret.curNode->w2 = this->val;
		this->curNode->operCount++;
		num.curNode->operCount++;

		return ret;
	}

	Num operator/(Num num)
	{
		Num ret = 0;
		ret.val = val / num.val;

		
		ret.curNode->first = curNode;
		ret.curNode->second = num.curNode;
		ret.curNode->w1 = 1/num.val;
		ret.curNode->w2 = val * (-1.0/ (num.val * num.val));
		curNode->operCount++;
		num.curNode->operCount++;

		return ret;
	}

};

std::ostream& operator<<(std::ostream& os, const Num& num)
{
	os << num.val;
	return os;
}

namespace AutoGradientSystem
{
	

	void CalcGrad(Node* final, double w)
	{
		//if (w == 0) std::cout << "Weights: " << w << "\n";
		final->operCount--;
		final->grad += w;

		if (final->operCount > 0) return;
		//sortedNodes.push_back(final);
		if (final->first) CalcGrad(final->first, final->grad * final->w1);
		if (final->second) CalcGrad(final->second, final->grad * final->w2);
	}

	void CalcGrad(Node* final)
	{
		CalcGrad(final, 1);
	}

	void ClearGrad(Node* final)
	{
		if (final->first) ClearGrad(final->first);
		if (final->second) ClearGrad(final->second);
		final->grad = 0;
	}

	void ClearEverything()
	{
		for (int i = 0; i < sortedNodes.size(); i++)
		{
			sortedNodes[i]->first = 0;
			sortedNodes[i]->second = 0;
			sortedNodes[i]->dontDestroyed = false;
			//sortedNodes[i]->grad = 0;
		}

		for (int i = 0; i < sortedNodes.size(); i++)
		{

			if (sortedNodes[i]->ownerCount <= 0 || sortedNodes[i]->beDed)
			{
				delete sortedNodes[i];
			}
			
		}
		sortedNodes.clear();
	}

};

namespace MathFunction
{
	int MaxIndex(const std::vector<Num>& inp)
	{
		int idx = 0;
		double val = -99999999999;
		for (int i = 0; i < inp.size(); i++)
		{
			if (val < inp[i].val)
			{
				val = inp[i].val;
				idx = i;
			}
		}
		return idx;
	}
	Num ReLu(Num x)
	{
		Num ret = 0;
		if (x.val >= 0)
		{
			ret.val = x.val;
			ret.curNode->first = x.curNode;
			ret.curNode->w1 = 1;
			x.curNode->operCount++;
		}
		else
		{
			ret.val = 0;
			ret.curNode->first = x.curNode;
			ret.curNode->w1 = 0;
			x.curNode->operCount++;
		}
		

		return ret;
	}
	

	Num Sigmoid(Num x)
	{
		Num ret = 0; 
		ret.val = 1.0 / (1.0 + exp(-x.val));
		ret.curNode->first = x.curNode;
		ret.curNode->w1 = ret.val * (1 - ret.val);
		x.curNode->operCount++;

		return ret;
	}

	Num Square(Num x)
	{
		Num ret = 0;
		ret.val = x.val * x.val;
		ret.curNode->first = x.curNode;
		ret.curNode->w1 = 2 * x.val;
		x.curNode->operCount++;

		return ret;
	}
}