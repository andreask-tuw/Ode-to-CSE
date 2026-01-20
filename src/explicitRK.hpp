#ifndef EXPLICITRK_HPP
#define EXPLICITRK_HPP

#include "timestepper.hpp"
#include <vector.hpp>
#include <matrix.hpp>
#include <stdexcept>
#include <cmath>

namespace ASC_ode {
  using namespace nanoblas;

  class ExplicitRungeKutta : public TimeStepper
  {
    Matrix<> m_a;
    Vector<> m_b, m_c;
    int m_stages;
    int m_dim;
    Vector<> m_k;
    Vector<> m_y_tmp;

  public:
    ExplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,
      const Matrix<> &a, const Vector<> &b, const Vector<> &c) 
    : TimeStepper(rhs), m_a(a), m_b(b), m_c(c),
    m_stages(static_cast<int>(c.size())),
    m_dim(static_cast<int>(rhs->dimX())),
    m_k(static_cast<size_t>(m_stages) * static_cast<size_t>(m_dim)),
    m_y_tmp(static_cast<size_t>(m_dim))
    {
      if (rhs->dimX() != rhs->dimF())
        throw std::invalid_argument("ExplicitRungeKutta requires dimX == dimF");

      if (a.rows() != c.size() || a.cols() != c.size())
        throw std::invalid_argument("ExplicitRungeKutta: A must be s x s with s=c.size()");
      if (b.size() != c.size())
        throw std::invalid_argument("ExplicitRungeKutta: b.size() must equal c.size()");

      // For explicit RK: A must be strictly lower triangular.
      // We allow tiny floating error tolerance.
      constexpr double tol = 1e-15;
      for (int i = 0; i < m_stages; i++)
      {
        for (int j = i; j < m_stages; j++)
        {
          if (std::abs(m_a(i, j)) > tol)
            throw std::invalid_argument("ExplicitRungeKutta: A must be strictly lower triangular (a_ij=0 for j>=i)");
        }
      }
    }

    void doStep(double tau, VectorView<double> y) override
    {
      for (int i = 0; i < m_stages; i++)
      {
        m_y_tmp = y;
        for (int j = 0; j < i; j++)
        {
           double val = m_a(i, j);
           if (val != 0.0)
             m_y_tmp += (tau * val) * m_k.range(static_cast<size_t>(j*m_dim), static_cast<size_t>((j+1)*m_dim));
        }
        
        auto k_i = m_k.range(static_cast<size_t>(i*m_dim), static_cast<size_t>((i+1)*m_dim));
        m_rhs->evaluate(m_y_tmp, k_i);
      }

      for (int j = 0; j < m_stages; j++)
        y += tau * m_b(j) * m_k.range(static_cast<size_t>(j*m_dim), static_cast<size_t>((j+1)*m_dim));
    }
  };


  // Convenience helper for classical 4th-order explicit RK tableau.
  // Returns (A,b,c) for use with ExplicitRungeKutta.
  inline auto ERK_RK4_Tableau()
  {
    Matrix<> a(4, 4);
    a = 0.0;
    a(1, 0) = 0.5;
    a(2, 1) = 0.5;
    a(3, 2) = 1.0;
    Vector<> b = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 };
    Vector<> c = { 0.0, 0.5, 0.5, 1.0 };
    return std::tuple{ a, b, c };
  }
}

#endif // EXPLICITRK_HPP