// ---- SD Model types ----

export interface ModelData {
  metadata: Record<string, unknown>;
  parameters: Record<string, number>;
  stocks: Record<string, StockSpec>;
  auxiliaries: Record<string, AuxiliarySpec>;
  subsystems: Record<string, SubsystemSpec>;
  agent_templates: Record<string, AgentTemplate>;
  stochastic: StochasticConfig;
}

export interface StockSpec {
  initial: number;
  units: string;
  inflows: string[];
  outflows: string[];
  subsystem: string;
}

export interface AuxiliarySpec {
  equation: string;
  operator: string;
  inputs: Record<string, string>;
  subsystem: string;
  units: string;
}

export interface SubsystemSpec {
  label: string;
  color: string;
}

// ---- Graph node types (for Cytoscape) ----

export interface GraphNode {
  name: string;
  kind: 'stock' | 'auxiliary' | 'agent_template';
  subsystem: string;
  operator?: string;
  inputs?: Record<string, string>;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  source: string;
  target: string;
  polarity: 'positive' | 'negative';
  role?: 'inflow' | 'outflow' | 'read' | 'write';
}

// ---- Agent types ----

export interface AgentTemplate {
  count: number;
  topology: 'all_to_all' | 'network' | 'spatial_1d' | 'spatial_2d';
  internal_stocks: Record<string, {
    initial: { distribution: string; mean?: number; sigma?: number; min?: number; max?: number };
  }>;
  decision_rules: DecisionRule[];
  reads: string[];
  writes: Record<string, { target: string; method: string; weight?: string }>;
  interactions: AgentInteraction[];
}

export interface DecisionRule {
  condition: string;
  action: string;
}

export interface AgentInteraction {
  type: 'competition' | 'cooperation';
  with: string;
  field: string;
  strength: number;
}

export interface AgentInstance {
  template: string;
  state: Record<string, number>;
  outputs: Record<string, number>;
}

// ---- Simulation types ----

export interface SimResult {
  id?: string;
  mode: 'deterministic' | 'stochastic';
  t?: number[];
  stocks?: Record<string, number[]>;
  auxiliaries?: Record<string, number[]>;
  ensemble?: {
    t: number[];
    stocks: Record<string, number[]>;
    auxiliaries: Record<string, number[]>;
  }[];
}

export interface StochasticConfig {
  sd_noise: Record<string, { noise_scale: number }>;
  agent_noise: { scale: number };
}

// ---- Loop analysis ----

export interface FeedbackLoop {
  nodes: string[];
  edges: [string, string, string][];
  polarity: 'reinforcing' | 'balancing';
}
