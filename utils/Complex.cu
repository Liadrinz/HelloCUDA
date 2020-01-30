#pragma once
#include <stdlib.h>
#include <math.h>
#include <iostream>

__device__ const double PI = 3.141592653589793238462643;

class Complex {
public:
    double real;
    double imag;

    // Wn
    __device__ static Complex W(int n) {
        Complex res(cos(2.0 * PI / n), sin(2.0 * PI / n));
        return res;
    }

    // Wn^k
    __device__ static Complex W(int n, int k) {
        Complex res(cos(2.0 * PI * k / n), sin(2.0 * PI * k / n));
        return res;
    }

    Complex() {

    }

    static Complex GetComplex(double real, double imag) {
        Complex r;
        r.real = real;
        r.imag = imag;
        return r;
    }

    static Complex GetRandomComplex() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = (double)rand() / rand();
        return r;
    }

    static Complex GetRandomReal() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = 0;
        return r;
    }

    static Complex GetRandomPureImag() {
        Complex r;
        r.real = 0;
        r.imag = (double)rand() / rand();
        return r;
    }

    __device__ Complex(double real, double imag) {
        this->real = real;
        this->imag = imag;
    }

    __device__ Complex operator+(const Complex &other) {
        Complex res(this->real + other.real, this->imag + other.imag);
        return res;
    }

    __device__ Complex operator-(const Complex &other) {
        Complex res(this->real - other.real, this->imag - other.imag);
        return res;
    }

    __device__ Complex operator*(const Complex &other) {
        Complex res(this->real * other.real - this->imag * other.imag, this->imag * other.real + this->real * other.imag);
        return res;
    }
};
