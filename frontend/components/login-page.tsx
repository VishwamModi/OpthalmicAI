'use client';

import { useState } from 'react';
import { Eye, Lock, Shield, EyeOff } from 'lucide-react';
import { apiUrl } from '@/lib/api-base';

export function LoginPage({ onNavigate }: { onNavigate: (view: string) => void }) {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [dob, setDob] = useState('');
  const [gender, setGender] = useState<'female' | 'male' | 'other' | 'prefer_not' | ''>('');
  const [patientNotes, setPatientNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const normalizedEmail = email.trim().toLowerCase();
  const normalizedPassword = password.replace(/\s+$/g, ''); // ignore accidental trailing spaces

  const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail);
  const passwordOk = normalizedPassword.length >= 8;
  const firstOk = !isSignUp || firstName.trim().length >= 2;
  const lastOk = !isSignUp || lastName.trim().length >= 2;
  const dobOk = !isSignUp || /^\d{4}-\d{2}-\d{2}$/.test(dob.trim());
  const genderOk = !isSignUp || Boolean(gender);
  const canSubmit = emailOk && passwordOk && firstOk && lastOk && dobOk && genderOk && !loading;

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!emailOk) {
      setError('Please enter a valid email address.');
      return;
    }
    if (!passwordOk) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (isSignUp && !firstOk) {
      setError('First name must be at least 2 characters.');
      return;
    }
    if (isSignUp && !lastOk) {
      setError('Last name must be at least 2 characters.');
      return;
    }
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const endpoint = isSignUp ? apiUrl('/api/signup') : apiUrl('/api/login');
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: normalizedEmail,
          password: normalizedPassword,
          first_name: firstName.trim(),
          last_name: lastName.trim(),
          dob: dob.trim() || null,
          gender: gender || null,
          patient_notes: patientNotes.trim() || null,
        }),
      });
      const data = (await res.json()) as any;
      if (!res.ok || data?.status !== 'success') {
        throw new Error(data?.message || 'Login failed');
      }
      if (isSignUp) {
        setSuccess('Patient profile created. Please sign in to continue.');
        setIsSignUp(false);
        setPassword('');
        setShowPassword(false);
        setDob('');
        setGender('');
        setPatientNotes('');
      } else {
        try {
          localStorage.setItem(
            'opthalmic_session',
            JSON.stringify({
              user: data?.user || null,
              patient_id: data?.patient_id || null,
              patient_uid: data?.patient_uid || null,
              patient_name: data?.patient_name || null,
              patient_dob: data?.patient_dob || null,
              doctor_name: data?.doctor_name || null,
            })
          );
        } catch {}
        onNavigate('dashboard');
      }
    } catch (err: any) {
      setError(err?.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#FCF8F3] flex flex-col">
      {/* Header */}
      <header className="border-b border-[#E8D9C8] bg-[#FCF8F3]">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => onNavigate('home')}
              className="inline-flex items-center gap-2 rounded-lg border border-[#D6C3AE] bg-white px-3 py-1.5 text-sm font-semibold text-[#3B2416] hover:bg-[#F4EADD] transition-colors"
            >
              ← Back to Home
            </button>
          </div>
        </div>
      </header>

      {/* Login Card */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <div className="bg-white rounded-xl border border-[#E8D9C8] shadow-sm p-8 mb-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-[#3B2416] mb-2">Patient Data Portal</h2>
              <p className="text-sm text-[#6E4B34]">
                Create and reuse your patient account to access reports
              </p>
            </div>

            <form onSubmit={handleLogin} className="space-y-4">
              {/* Sign-up expanded fields */}
              <div
                className={[
                  'overflow-hidden transition-all duration-300',
                  isSignUp ? 'max-h-[520px] opacity-100' : 'max-h-0 opacity-0',
                ].join(' ')}
              >
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
                  <div>
                    <label className="block text-sm font-medium text-[#111111] mb-2">
                      First Name
                    </label>
                    <input
                      type="text"
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      placeholder="First name"
                      className="w-full px-4 py-2 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] placeholder-[#8A8A8A] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60"
                      required={isSignUp}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#111111] mb-2">
                      Last Name
                    </label>
                    <input
                      type="text"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      placeholder="Last name"
                      className="w-full px-4 py-2 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] placeholder-[#8A8A8A] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60"
                      required={isSignUp}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="block text-sm font-medium text-[#111111] mb-2">
                      Date of Birth
                    </label>
                    <input
                      type="date"
                      value={dob}
                      onChange={(e) => setDob(e.target.value)}
                      className="w-full px-4 py-2 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60"
                      required={isSignUp}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#111111] mb-2">
                      Gender
                    </label>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value as any)}
                      className="w-full px-4 py-2 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60"
                      required={isSignUp}
                    >
                      <option value="" disabled>
                        Select…
                      </option>
                      <option value="female">Female</option>
                      <option value="male">Male</option>
                      <option value="other">Other</option>
                      <option value="prefer_not">Prefer not to say</option>
                    </select>
                  </div>
                </div>

                <div className="mt-4">
                  <label className="block text-sm font-medium text-[#111111] mb-2">
                    Notes (optional)
                  </label>
                  <textarea
                    value={patientNotes}
                    onChange={(e) => setPatientNotes(e.target.value)}
                    rows={3}
                    placeholder="Any relevant history (optional)"
                    className="w-full px-4 py-2 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] placeholder-[#8A8A8A] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60 resize-none"
                  />
                </div>
              </div>

              {/* Email Field */}
              <div>
                <label className="block text-sm font-medium text-[#111111] mb-2">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@example.com"
                  className="w-full px-4 py-2 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] placeholder-[#8A8A8A] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60"
                  required
                />
              </div>

              {/* Password Field */}
              <div>
                <label className="block text-sm font-medium text-[#111111] mb-2">
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full px-4 py-2 pr-11 border border-[#DCDCDC] rounded-lg bg-white text-[#111111] placeholder-[#8A8A8A] transition-colors focus:outline-none focus:ring-0 focus:border-[#111111]/60"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    className="absolute inset-y-0 right-2 inline-flex items-center justify-center w-9 rounded-md text-[#666666] hover:text-[#111111] hover:bg-[#F3F3F3] transition-colors"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              <button
                type="submit"
                disabled={!canSubmit}
                className="w-full bg-[#3B2416] hover:bg-[#2F1B11] disabled:opacity-50 text-white font-semibold py-2 rounded-lg transition-colors mt-6"
              >
                {loading ? (isSignUp ? 'Creating patient profile…' : 'Signing in…') : isSignUp ? 'Create Patient Profile' : 'Sign In'}
              </button>

              {/* Bottom toggle */}
              {!isSignUp ? (
                <button
                  type="button"
                  onClick={() => setIsSignUp(true)}
                  className="w-full text-center text-xs text-[#111111] underline underline-offset-4 hover:text-black transition-colors"
                >
                  Don't have an account? Sign up
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setIsSignUp(false)}
                  className="w-full text-center text-xs text-[#111111] underline underline-offset-4 hover:text-black transition-colors"
                >
                  Already have an account? Sign in
                </button>
              )}
            </form>

            {error && (
              <div className="mt-4 rounded-lg border border-[#E5B7B7] bg-[#FFF5F5] px-3 py-2 text-sm text-[#9F2A2A]">
                {error}
              </div>
            )}
            {success && (
              <div className="mt-4 rounded-lg border border-[#D4CCC2] bg-[#F0F4EE] px-3 py-2 text-sm text-[#6B705C]">
                {success}
              </div>
            )}

            {/* Trust Badges */}
            <div className="mt-8 pt-8 border-t border-[#E5E5E5] space-y-3">
              <div className="flex items-center gap-3">
                <Shield className="w-4 h-4 text-[#111111]" />
                <span className="text-sm text-[#111111]">Secure authentication</span>
              </div>
              <div className="flex items-center gap-3">
                <Lock className="w-4 h-4 text-[#111111]" />
                <span className="text-sm text-[#111111]">Encrypted transport</span>
              </div>
            </div>
          </div>

          {/* Demo Credentials */}
          <div className="bg-[#F8F2EA] border border-[#E8D9C8] rounded-lg p-4 text-center">
            <p className="text-xs text-[#6E4B34]">
              <strong>Tip:</strong>
              <br />
              Sign up once with your own email.
              <br />
              Then sign in anytime with the same credentials.
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-[#E8D9C8] bg-white py-6 px-6">
        <div className="mx-auto max-w-6xl text-center text-xs text-[#6E4B34]">
          <p>© 2026 Opthalmic. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
