#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <array>
#include <ostream>
#include <cstddef>



using namespace ASC_ode;

#include <vector.hpp>
using namespace nanoblas;


template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};


template <int D>
class Fix
{
public:
  Vec<D> pos;
};


class Connector
{
public:
  enum CONTYPE { FIX=1, MASS=2 };
  CONTYPE type;
  size_t nr;
};

std::ostream & operator<< (std::ostream & ost, const Connector & con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;  
  double stiffness;
  std::array<Connector,2> connectors;
};


// adding distance constraint between two connectors - here only storing the data
class DistanceConstraint
{
public:
  std::array<Connector,2> connectors;
  double distance;
  DistanceConstraint (Connector c1, Connector c2, double dist)
    : connectors{{c1,c2}}, distance(dist) { }
};


template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  std::vector<DistanceConstraint> m_joints;
  Vec<D> m_gravity=0.0;
public:
  void setGravity (Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix (Fix<D> p)
  {
    m_fixes.push_back(p);
    return { Connector::FIX, m_fixes.size()-1 };
  }

  Connector addMass (Mass<D> m)
  {
    m_masses.push_back (m);
    return { Connector::MASS, m_masses.size()-1 };
  }
  
  size_t addSpring (Spring s) 
  {
    m_springs.push_back (s); 
    return m_springs.size()-1;
  }

  size_t addJoint (DistanceConstraint j)
  {
    m_joints.push_back (j);
    return m_joints.size()-1;
  }

  auto & fixes() { return m_fixes; } 
  auto & masses() { return m_masses; } 
  auto & springs() { return m_springs; }
  auto & joints() { return m_joints; }

  void getState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        valmat.row(i) = m_masses[i].pos;
        dvalmat.row(i) = m_masses[i].vel;
        ddvalmat.row(i) = m_masses[i].acc;
      }
  }

  void setState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        m_masses[i].pos = valmat.row(i);
        m_masses[i].vel = dvalmat.row(i);
        m_masses[i].acc = ddvalmat.row(i);
      }
  }
};

template <int D>
std::ostream & operator<< (std::ostream & ost, MassSpringSystem<D> & mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;

  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;

  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;
  return ost;
}


template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
public:
  MSS_Function (MassSpringSystem<D> & _mss)
    : mss(_mss) { }

  virtual size_t dimX() const override { return D*mss.masses().size() + mss.joints().size(); }
  virtual size_t dimF() const override { return D*mss.masses().size() + mss.joints().size(); }


  virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f = 0.0;

    const size_t nm = mss.masses().size();
    const size_t nj = mss.joints().size();

    // x = [positions | lambdas]
    auto xmat    = x.range(0, nm*D).asMatrix(nm, D);
    auto lambdas = x.range(nm*D, nm*D + nj);

    // f = [accelerations | constraint residuals]
    auto fmat = f.range(0, nm*D).asMatrix(nm, D);
    auto gres = f.range(nm*D, nm*D + nj);

    // gravity (force)
    for (size_t i = 0; i < nm; i++)
      fmat.row(i) = mss.masses()[i].mass * mss.getGravity();

    // spring forces
    for (auto spring : mss.springs())
    {
      auto [c1,c2] = spring.connectors;

      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);

      Vec<D> d = p2 - p1;
      double L = norm(d);
      if (L == 0.0) continue;

      Vec<D> dir12 = (1.0 / L) * d;
      double force = spring.stiffness * (L - spring.length);

      if (c1.type == Connector::MASS) fmat.row(c1.nr) += force * dir12;
      if (c2.type == Connector::MASS) fmat.row(c2.nr) -= force * dir12;
    }

    // constraint forces + residuals
    for (size_t j = 0; j < nj; j++)
    {
      auto joint = mss.joints()[j];
      auto [c1,c2] = joint.connectors;

      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);

      Vec<D> diff = p1 - p2;
      double lambda = lambdas(j);

      // Lagrange multiplier force: ± 2 λ (p1 - p2)
      if (c1.type == Connector::MASS) fmat.row(c1.nr) += (2.0 * lambda) * diff;
      if (c2.type == Connector::MASS) fmat.row(c2.nr) -= (2.0 * lambda) * diff;

      // constraint residual: |p1-p2|^2 - d^2
      gres(j) = dot(diff, diff) - joint.distance * joint.distance;
    }

    // forces -> accelerations
    for (size_t i = 0; i < nm; i++)
      fmat.row(i) *= 1.0 / mss.masses()[i].mass;
  }


virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    const size_t nm = mss.masses().size();
    const size_t nj = mss.joints().size();
    const size_t n  = D * nm + nj;

    // zero df
    for (size_t i = 0; i < df.rows(); i++)
      for (size_t j = 0; j < df.cols(); j++)
        df(i,j) = 0.0;

    auto xmat = x.range(0, D * nm).asMatrix(nm, D);

    // Springs contribution
    for (auto spring : mss.springs())
    {
      auto [c1, c2] = spring.connectors;

      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);

      Vec<D> diff = p1 - p2;
      double L = norm(diff);
      if (L == 0.0) continue;

      // unit direction from p2 -> p1
      Vec<D> dir = diff;
      for (size_t k = 0; k < D; k++)
        dir(k) /= L;

      // K = k * (I - dir*dir^T)
      for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++)
        {
          double val = spring.stiffness * ((a == b ? 1.0 : 0.0) - dir(a) * dir(b));

          if (c1.type == Connector::MASS && c2.type == Connector::MASS)
          {
            df(D * c1.nr + a, D * c1.nr + b) += val;
            df(D * c1.nr + a, D * c2.nr + b) -= val;
            df(D * c2.nr + a, D * c1.nr + b) -= val;
            df(D * c2.nr + a, D * c2.nr + b) += val;
          }
          else if (c1.type == Connector::MASS && c2.type == Connector::FIX)
          {
            df(D * c1.nr + a, D * c1.nr + b) += val;
          }
          else if (c1.type == Connector::FIX && c2.type == Connector::MASS)
          {
            df(D * c2.nr + a, D * c2.nr + b) += val;
          }
        }
    }

    // Distance constraints (your m_joints) contribution
    for (size_t i = 0; i < nj; i++)
    {
      auto & joint = mss.joints()[i];
      auto [c1, c2] = joint.connectors;

      Vec<D> p1 = (c1.type == Connector::FIX) ? mss.fixes()[c1.nr].pos : xmat.row(c1.nr);
      Vec<D> p2 = (c2.type == Connector::FIX) ? mss.fixes()[c2.nr].pos : xmat.row(c2.nr);

      Vec<D> diff = p1 - p2;

      // Column index of lambda_i in x:
      const size_t col_lambda = D * nm + i;
      // Row index of residual g_i in f:
      const size_t row_g      = D * nm + i;

      // derivatives involving lambda in force rows:
      for (int a = 0; a < D; a++)
      {
        if (c1.type == Connector::MASS)
          df(D * c1.nr + a, col_lambda) += 2.0 * diff(a);
        if (c2.type == Connector::MASS)
          df(D * c2.nr + a, col_lambda) -= 2.0 * diff(a);
      }

      // derivatives of residual g = ||p1-p2||^2 - d^2 w.r.t positions
      for (int a = 0; a < D; a++)
      {
        if (c1.type == Connector::MASS)
          df(row_g, D * c1.nr + a) += 2.0 * diff(a);
        if (c2.type == Connector::MASS)
          df(row_g, D * c2.nr + a) -= 2.0 * diff(a);
      }

      // Note: dg/dlambda = 0 (so df(row_g, col_lambda) stays 0)
    }

    // Convert force-rows -> acceleration-rows by dividing by mass
    for (size_t i = 0; i < nm; i++)
    {
      const double m = mss.masses()[i].mass;
      for (size_t a = 0; a < D; a++)
        for (size_t j = 0; j < n; j++)
          df(D * i + a, j) /= m;
    }
  }

  
};

#endif
