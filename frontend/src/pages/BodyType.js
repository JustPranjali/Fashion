import React, { useState } from 'react';
import axios from 'axios';
import useScrollAnimations from '../components/ScrollAnimations';
import '../styles/ScrollAnimation.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const BodyType = () => {
  useScrollAnimations(); // <-- activates your IntersectionObserver

  const [form, setForm] = useState({
    height: '',
    weight: '',
    bust: '',
    cup: '',
    waist: '',
    hip: '',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const update = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/bodytype`, {
        height: form.height,
        weight: form.weight,
        bust: form.bust,
        cup: form.cup,
        waist: form.waist,
        hip: form.hip,
      });
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.error || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page page--warm">
      {/* Hero */}
      <section className="hero hidden-initially animate-delay-100">
        <div className="hero__overlay" />
        <div className="hero__content">
          <h1 className="hero__title">
            Discover Your <span className="hero__title--accent">Body Type</span>
          </h1>
          <p className="hero__subtitle">
            Enter your measurements to get your BMI, shape, and overall body type—styled to
            match the rest of StyleMate.
          </p>
          <a href="#calculator" className="btn btn--lg btn--primary">
            Start
          </a>
        </div>
      </section>

      {/* Calculator */}
      <section id="calculator" className="section section--padded">
        <div className="container grid-2">
          {/* Form */}
          <form onSubmit={handleSubmit} className="card hidden-initially animate-delay-150">
            <h2 className="card__title">Body Type Calculator</h2>
            <p className="card__subtitle">We don’t store your inputs. Results are for guidance only.</p>

            <div className="grid-2 grid-gap">
              <div className="form__group">
                <label className="form__label">Height (m)</label>
                <input
                  name="height"
                  value={form.height}
                  onChange={update}
                  placeholder="e.g., 1.70"
                  className="input"
                />
              </div>
              <div className="form__group">
                <label className="form__label">Weight (kg)</label>
                <input
                  name="weight"
                  value={form.weight}
                  onChange={update}
                  placeholder="e.g., 70"
                  className="input"
                />
              </div>

              <div className="form__group">
                <label className="form__label">Bust (in)</label>
                <input
                  name="bust"
                  value={form.bust}
                  onChange={update}
                  placeholder="e.g., 36"
                  className="input"
                />
              </div>
              <div className="form__group">
                <label className="form__label">Cup</label>
                <input
                  name="cup"
                  value={form.cup}
                  onChange={update}
                  placeholder="e.g., D"
                  className="input input--upper"
                />
              </div>

              <div className="form__group">
                <label className="form__label">Waist (in)</label>
                <input
                  name="waist"
                  value={form.waist}
                  onChange={update}
                  placeholder="e.g., 28"
                  className="input"
                />
              </div>
              <div className="form__group">
                <label className="form__label">Hip (in)</label>
                <input
                  name="hip"
                  value={form.hip}
                  onChange={update}
                  placeholder="e.g., 38"
                  className="input"
                />
              </div>
            </div>

            <button type="submit" className="btn btn--primary btn--full" disabled={loading}>
              {loading ? 'Calculating…' : 'Calculate'}
            </button>

            {error && <div className="alert alert--error">{error}</div>}
          </form>

          {/* Results */}
          <div className="card hidden-initially animate-delay-250">
            <h2 className="card__title">Results</h2>
            {!result && !error && <p className="muted">Results will appear here.</p>}

            {result && (
              <div className="result-grid">
                <div className="result-tile">
                  <div className="result-tile__label">BMI</div>
                  <div className="result-tile__value">{result.bmi}</div>
                </div>
                <div className="result-tile">
                  <div className="result-tile__label">Breast</div>
                  <div className="result-tile__value">
                    {result.breast_desc} <span className="muted">({result.breast_multiplier})</span>
                  </div>
                </div>
                <div className="result-tile">
                  <div className="result-tile__label">Butt</div>
                  <div className="result-tile__value">{result.butt_desc}</div>
                </div>
                <div className="result-tile">
                  <div className="result-tile__label">Body Shape</div>
                  <div className="result-tile__value">{result.body_shape}</div>
                </div>
                <div className="result-tile">
                  <div className="result-tile__label">Body Type</div>
                  <div className="result-tile__value">{result.body_type}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

export default BodyType;
