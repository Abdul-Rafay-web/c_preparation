export type ModelKey = 'fp32' | 'qat_8b_p50' | 'qat_6b_p75' | 'qat_4b_p50';

export interface LayerWeights {
  in: number;
  out: number;
  w: number[][];
  b: number[];
  nonzero: number;
  total: number;
}

export interface ExportedModel {
  name: string;
  bits: number;
  w_int: number;
  a_bits: number;
  a_int: number;
  sparsity_pct: number;
  bops: number;
  bop_reduction_pct: number;
  test_acc: number;
  macro_auc: number;
  rej_50: number;
  layers: LayerWeights[];
}

export interface SampleJet {
  id: string;
  true_class_idx: number;
  true_class: 'g' | 'q' | 'w' | 'z' | 't';
  true_label: string;
  difficulty: 'Textbook' | 'Borderline';
  mass_gev: number;
  multiplicity: number;
  c2_b1: number;
  d2_b1: number;
  x_scaled: number[];
  x_raw: number[];
}

export interface InferenceBundle {
  feature_names: string[];
  class_names: ('g' | 'q' | 'w' | 'z' | 't')[];
  class_labels: Record<string, string>;
  models: Record<ModelKey, ExportedModel>;
  sample_jets: SampleJet[];
}

export interface Level1Point {
  tag: string;
  weight_bits: number;
  target_sparsity: number;
  actual_sparsity: number;
  nonzero_weights: number;
  weight_kb: number;
  effective_bops: number;
  bop_reduction_pct: number;
  accuracy: number;
  macro_auc: number;
  mean_rej_50: number;
  mean_rej_80: number;
}

export interface Level2Point {
  tag: string;
  total_bits: number;
  target_sparsity: number;
  actual_sparsity: number;
  pquant_ebops: number;
  ebops_reduction_pct: number;
  accuracy: number;
  macro_auc: number;
  mean_rej_50: number;
  mean_rej_80: number;
}

export interface RocPoint {
  tpr: number;
  fpr: number;
}

export interface DashboardData {
  ptq_vs_qat: {
    bits: number;
    label: string;
    ptq_acc: number;
    qat_acc: number;
    ptq_auc: number;
    qat_auc: number;
  }[];
  level1_sweep: Level1Point[];
  level2_sweep: {
    data_mode: string;
    n_constituents: number;
    results: Level2Point[];
  };
  hls_verification: {
    Model: string;
    'PyTorch AUC': number;
    ΔAUC: number;
    'Top-1 Agreement (%)': number;
    'Mean |Δlogit|': number;
  }[];
  roc_curves: Record<
    string,
    {
      label: string;
      fp32_auc: number;
      q6_auc: number;
      fp32_rej50: number;
      q6_rej50: number;
      fp32: RocPoint[];
      q6: RocPoint[];
    }
  >;
  firmware: {
    defines_h: string;
    project_cpp: string;
    w2_sample: number[];
    w2_zero_pct: number;
    w2_total: number;
  };
}
