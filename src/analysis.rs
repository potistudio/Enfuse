use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

#[derive(Clone, Debug, Default)]
pub struct WaveformPoint {
	pub amplitude: f32,
	pub color: [f32; 3], // {R, G, B}
}

pub fn analyze_waveform(samples: &[f32], sample_rate: u32) -> Vec<WaveformPoint> {
	let mut planner = FftPlanner::new();
	let fft_window = 4096; // Smaller window for better time resolution (~10.7Hz per bin @ 44.1kHz)
	let hop_size = fft_window / 2; // 50% overlap for smoother transitions
	let stride = 64; // Output resolution

	let fft = planner.plan_fft_forward(fft_window);
	let mut scratch = vec![Complex::new(0.0, 0.0); fft_window];

	// Pre-compute Hanning window
	let hanning: Vec<f32> = (0..fft_window)
		.map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_window as f32 - 1.0)).cos()))
		.collect();

	let bin_res = sample_rate as f32 / fft_window as f32;
	let min_bin = (20.0 / bin_res).ceil() as usize;
	let max_bin = (fft_window / 2 - 1).min((20000.0 / bin_res).floor() as usize);

	let mut points = Vec::new();

	// STFT with overlap
	let mut window_start = 0usize;
	while window_start + fft_window <= samples.len() {
		let chunk = &samples[window_start..window_start + fft_window];

		// Apply Hanning window and prepare FFT buffer
		let mut buffer: Vec<Complex<f32>> = chunk
			.iter()
			.zip(hanning.iter())
			.map(|(&s, &w)| Complex::new(s * w, 0.0))
			.collect();

		fft.process_with_scratch(&mut buffer, &mut scratch);

		// Find peak bin
		let mut peak_mag = 0.0f32;
		let mut peak_bin = min_bin;
		for i in min_bin..=max_bin {
			let mag = buffer[i].norm();
			if mag > peak_mag {
				peak_mag = mag;
				peak_bin = i;
			}
		}

		// Calculate local spectral centroid around peak (±30 bins)
		let local_range = 30usize;
		let local_min = peak_bin.saturating_sub(local_range).max(1);
		let local_max = (peak_bin + local_range).min(fft_window / 2 - 1);

		let mut weighted_sum = 0.0f32;
		let mut magnitude_sum = 0.0f32;

		for i in local_min..=local_max {
			let mag = buffer[i].norm();
			let freq = i as f32 * bin_res;
			weighted_sum += freq * mag;
			magnitude_sum += mag;
		}

		let centroid_freq = if magnitude_sum > 0.0 {
			weighted_sum / magnitude_sum
		} else {
			peak_bin as f32 * bin_res
		};

		// Map frequency to Hue using log scale (60Hz - 5kHz → 0° - 270°)
		let freq_log = centroid_freq.max(60.0).min(5000.0).log2();
		let normalized = (freq_log - 5.91) / (12.29 - 5.91); // 0.0 to 1.0
		let hue = normalized.clamp(0.0, 1.0) * 270.0;

		// HSV to RGB (S=1.0, V=1.0)
		let color = {
			let h = hue / 60.0;
			let i = h.floor() as i32;
			let f = h - i as f32;
			let q = 1.0 - f;
			let t = f;

			match i % 6 {
				0 => [1.0, t, 0.0],
				1 => [q, 1.0, 0.0],
				2 => [0.0, 1.0, t],
				3 => [0.0, q, 1.0],
				4 => [t, 0.0, 1.0],
				_ => [1.0, 0.0, q],
			}
		};

		// Generate points for this window's hop region
		let hop_end = (window_start + hop_size).min(samples.len());
		for i in (window_start..hop_end).step_by(stride) {
			if i < samples.len() {
				points.push(WaveformPoint {
					amplitude: samples[i],
					color,
				});
			}
		}

		window_start += hop_size;
	}

	// Handle remaining samples at the end
	let remaining_start = window_start;
	if remaining_start < samples.len() {
		// Use last computed color or default
		let default_color = [0.5, 0.5, 0.5];
		let last_color = points.last().map(|p| p.color).unwrap_or(default_color);

		for i in (remaining_start..samples.len()).step_by(stride) {
			points.push(WaveformPoint {
				amplitude: samples[i],
				color: last_color,
			});
		}
	}

	points
}

/// BPM検出の結果を保持する構造体
#[derive(Clone, Debug, Default)]
pub struct BpmResult {
	pub bpm: f32,
	pub beat_offset: f32, // 秒単位での最初の拍の位置
}

pub fn detect_bpm(samples: &[f32], sample_rate: u32, channels: u16) -> BpmResult {
	// BeatNet offline path (LOG_SPECT → CRNN → beat decode). Spectral-flux is
	// fallback only — if BeatNet mis-detects, fix feature/model fidelity, don't
	// prefer the procedural estimator.
	if let Some(result) = crate::beatnet::detect_bpm(samples, sample_rate, channels) {
		return result;
	}
	detect_bpm_spectral_flux(samples, sample_rate)
}

fn detect_bpm_spectral_flux(samples: &[f32], sample_rate: u32) -> BpmResult {
	let window_size = 2048;
	let hop_size = 512;

	// 1. STFT & Onset Envelope
	// プランナーは重いので外で作るか、構造体で保持するのがベストですが、ここでは関数内で一度だけ作成
	let mut planner = FftPlanner::new();
	let onset_env = calculate_onset_envelope(samples, window_size, hop_size, &mut planner);

	// 2. Tempogram / Periodicity (FFT optimized)
	let (tempo, _confidence) = estimate_tempo(&onset_env, sample_rate, hop_size, &mut planner);

	// 3. Beat Offset Detection (最初のダウンビート位置を検出)
	let beat_offset = if tempo > 0.0 && !onset_env.is_empty() {
		detect_first_beat(&onset_env, tempo, sample_rate, hop_size)
	} else {
		0.0
	};

	BpmResult {
		bpm: tempo,
		beat_offset,
	}
}

/// onset envelopeから最初のダウンビート位置を検出する
fn detect_first_beat(onset_env: &[f32], bpm: f32, sample_rate: u32, hop_size: usize) -> f32 {
	if onset_env.is_empty() || bpm <= 0.0 {
		return 0.0;
	}

	let env_sr = sample_rate as f32 / hop_size as f32; // onset envelopeのサンプルレート
	let frames_per_beat = env_sr * 60.0 / bpm; // 1拍あたりのフレーム数

	// 最初の2拍分の範囲で最も強いonsetを探す
	let search_range = (frames_per_beat * 2.0).ceil() as usize;
	let search_range = search_range.min(onset_env.len());

	// 最大のonsetピークを見つける
	let first_beat_frame = onset_env[..search_range]
		.iter()
		.enumerate()
		.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
		.map(|(i, _)| i)
		.unwrap_or(0);

	// フレームから秒に変換
	first_beat_frame as f32 * hop_size as f32 / sample_rate as f32
}

fn calculate_onset_envelope(
	samples: &[f32],
	window_size: usize,
	hop_size: usize,
	planner: &mut FftPlanner<f32>,
) -> Vec<f32> {
	let fft = planner.plan_fft_forward(window_size);
	let mut envelope = Vec::with_capacity(samples.len() / hop_size);

	// スペクトルの差分計算用バッファ
	let mut prev_spectrum = vec![0.0; window_size / 2 + 1];

	// FFT用バッファ（再利用する）
	let mut buffer = vec![Complex::new(0.0, 0.0); window_size];
	let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

	// Hanning Window (事前に計算)
	let window: Vec<f32> = (0..window_size)
		.map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (window_size as f32 - 1.0)).cos()))
		.collect();

	// ループ処理
	// bounds checkを減らすために windows を使う手もありますが、
	// hop_size があるため step_by で実装
	let max_start = samples.len().saturating_sub(window_size);

	for chunk_start in (0..max_start).step_by(hop_size) {
		let chunk = &samples[chunk_start..chunk_start + window_size];

		// 窓関数を適用してバッファにコピー
		for (i, (&s, &w)) in chunk.iter().zip(&window).enumerate() {
			buffer[i] = Complex::new(s * w, 0.0);
		}

		// FFT実行
		fft.process_with_scratch(&mut buffer, &mut scratch);

		// Spectral Flux 計算
		let mut flux = 0.0;
		// ナイキスト周波数まで
		for i in 0..window_size / 2 + 1 {
			let mag = buffer[i].norm();
			// Log compression (log(1 + mag)) to simulate human hearing
			let log_mag = (1.0 + mag).ln();

			// Half-Wave Rectification (正の差分のみ採用＝音の立ち上がり)
			let diff = log_mag - prev_spectrum[i];
			if diff > 0.0 {
				flux += diff;
			}
			prev_spectrum[i] = log_mag;
		}
		envelope.push(flux);
	}

	envelope
}

fn estimate_tempo(onset_env: &[f32], sample_rate: u32, hop_size: usize, planner: &mut FftPlanner<f32>) -> (f32, f32) {
	let env_sr = sample_rate as f32 / hop_size as f32;
	let n = onset_env.len();

	// パディング：循環畳み込みを避けるため、長さの2倍以上の2のべき乗にする
	let padded_len = (n * 2).next_power_of_two();

	let fft = planner.plan_fft_forward(padded_len);
	let ifft = planner.plan_fft_inverse(padded_len);

	// 1. FFTによる自己相関の計算
	let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(padded_len);

	// データをコピー (平均を引いてDC成分を除去するとより精度が出ますが、今回はそのまま)
	for &x in onset_env {
		buffer.push(Complex::new(x, 0.0));
	}
	// ゼロパディング
	buffer.resize(padded_len, Complex::new(0.0, 0.0));

	// Forward FFT
	fft.process(&mut buffer);

	// Power Spectrum ( Magnitude Squared )
	for c in buffer.iter_mut() {
		// |X[k]|^2  (共役を掛けるのと同義)
		let mag_sq = c.norm_sqr();
		*c = Complex::new(mag_sq, 0.0);
	}

	// Inverse FFT (これにより自己相関関数が得られる)
	ifft.process(&mut buffer);

	// 2. ピーク検出とBPM推定
	let min_bpm = 40.0;
	let max_bpm = 220.0;

	let max_lag = (env_sr * 60.0 / min_bpm) as usize;
	let min_lag = (env_sr * 60.0 / max_bpm) as usize;

	let mut max_corr = 0.0;
	let mut best_lag = 0.0; // 補間後の正確なラグ

	// スケールファクタ（IFFTの結果はサイズ倍されているため）
	// 正規化しないと数値が大きくなりすぎるが、比較だけなら不要。
	// しかしバイアス補正のために必要。
	let scale = 1.0 / padded_len as f32;

	for lag in min_lag..=max_lag.min(n / 2) {
		if lag >= buffer.len() {
			break;
		}

		let raw_corr = buffer[lag].re * scale;

		// Bias correction: longer lags overlap fewer samples
		let divisor = (n - lag) as f32;
		let normalized_corr = if divisor > 0.0 { raw_corr / divisor } else { 0.0 };

		if normalized_corr > max_corr {
			max_corr = normalized_corr;
			best_lag = lag as f32;
		}
	}

	if best_lag == 0.0 {
		return (0.0, 0.0);
	}

	// Resolve common octave / 3:2 confusions without a 120-BPM prior.
	best_lag = resolve_tempo_lag(&buffer, best_lag, min_lag, max_lag.min(n / 2), scale, n, env_sr);

	// 3. 放物線補間 (Parabolic Interpolation) によるサブフレーム精度の向上
	// ピークの前後を使って、真のピーク位置を推定する
	let idx = best_lag.round() as usize;
	if idx > 0 && idx < buffer.len() - 1 {
		let y_alpha = buffer[idx - 1].re; // 前
		let y_beta = buffer[idx].re; // 現在
		let y_gamma = buffer[idx + 1].re; // 次

		// 簡易的な補間式
		let denominator = 2.0 * (2.0 * y_beta - y_gamma - y_alpha);
		if denominator.abs() > 1e-5 {
			let delta = (y_gamma - y_alpha) / denominator;
			best_lag = idx as f32 + delta;
		}
	}

	let mut bpm = 60.0 * env_sr / best_lag;

	// 最も近い整数
	let nearest_int = bpm.round();

	// 整数との差分（絶対値）
	let diff = (bpm - nearest_int).abs();

	// 【設定】吸着の強さ (閾値)
	// 0.25 〜 0.3 くらいが適切です。
	// 例: 127.8 -> 差分0.2 -> 整数(128.0)採用
	// 例: 127.5 -> 差分0.5 -> そのまま(127.5)採用
	let snap_threshold = 0.3;

	if diff < snap_threshold {
		bpm = nearest_int;
	}

	(bpm, max_corr)
}

/// Among octave / 3:2 relatives of `best_lag`, pick the lag with highest
/// normalized autocorrelation that still looks like a local peak.
fn resolve_tempo_lag(
	buffer: &[Complex<f32>],
	best_lag: f32,
	min_lag: usize,
	max_lag: usize,
	scale: f32,
	n: usize,
	env_sr: f32,
) -> f32 {
	let lag_corr = |lag: usize| -> f32 {
		if lag == 0 || lag >= buffer.len() || lag > n / 2 {
			return f32::NEG_INFINITY;
		}
		let divisor = (n - lag) as f32;
		if divisor <= 0.0 {
			return f32::NEG_INFINITY;
		}
		buffer[lag].re * scale / divisor
	};

	let base = best_lag.round().max(1.0) as usize;
	let base_c = lag_corr(base);
	let factors: [(f32, f32); 5] = [
		(1.0, 1.0),
		(1.0, 2.0), // half lag → double BPM
		(2.0, 1.0), // double lag → half BPM
		(2.0, 3.0), // 2/3 lag → 3/2 BPM (150↔100 style)
		(3.0, 2.0), // 3/2 lag → 2/3 BPM
	];

	let mut best = base;
	let mut best_score = {
		let bpm = 60.0 * env_sr / base as f32;
		let bonus = if (70.0..=180.0).contains(&bpm) {
			0.02 * base_c.abs()
		} else {
			0.0
		};
		base_c + bonus
	};

	for (num, den) in factors {
		let lag = ((base as f32) * num / den).round() as usize;
		if lag < min_lag || lag > max_lag || lag == base {
			continue;
		}
		// Require a local peak so we don't slide onto a shoulder.
		let c = lag_corr(lag);
		let left = lag_corr(lag.saturating_sub(1));
		let right = lag_corr(lag + 1);
		if c < left || c < right {
			continue;
		}
		let bpm = 60.0 * env_sr / lag as f32;
		let band_bonus = if (70.0..=180.0).contains(&bpm) {
			0.02 * c.abs()
		} else {
			0.0
		};
		let score = c + band_bonus;
		if score > best_score {
			best_score = score;
			best = lag;
		}
	}

	best as f32
}

pub fn compute_spectrum(samples: &[f32]) -> Vec<f32> {
	let mut planner = FftPlanner::new();
	let window_size = samples.len();
	if window_size == 0 {
		return Vec::new();
	}
	let fft = planner.plan_fft_forward(window_size);

	// Hanning Window
	let window: Vec<f32> = (0..window_size)
		.map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos()))
		.collect();

	let mut buffer: Vec<Complex<f32>> = samples
		.iter()
		.zip(window.iter())
		.map(|(&s, &w)| Complex::new(s * w, 0.0))
		.collect();

	// Pad if needed (though we expect exact size usually)
	if buffer.len() < window_size {
		buffer.resize(window_size, Complex::new(0.0, 0.0));
	}

	let mut scratch = vec![Complex::new(0.0, 0.0); window_size];
	fft.process_with_scratch(&mut buffer, &mut scratch);

	// Return magnitudes (half spectrum)
	buffer.iter().take(window_size / 2).map(|c| c.norm()).collect()
}
