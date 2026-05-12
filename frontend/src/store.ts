import { create } from 'zustand';
import type { ModelData, SimResult, GraphNode, GraphEdge, AgentInstance } from './types';

interface AppState {
  // Model
  model: ModelData | null;
  setModel: (model: ModelData) => void;

  // Graph (derived from model, but positions are UI-only)
  nodes: GraphNode[];
  edges: GraphEdge[];
  setGraph: (nodes: GraphNode[], edges: GraphEdge[]) => void;

  // Simulation
  simResult: SimResult | null;
  setSimResult: (result: SimResult | null) => void;
  simRunning: boolean;
  setSimRunning: (running: boolean) => void;

  // Agents
  agentPopulation: AgentInstance[];
  setAgentPopulation: (agents: AgentInstance[]) => void;

  // UI state
  showAgentBridge: boolean;
  setShowAgentBridge: (show: boolean) => void;
  selectedTimestep: number;
  setSelectedTimestep: (t: number) => void;

  // Derived: is there a result to display
  hasResult: () => boolean;
}

export const useStore = create<AppState>((set, get) => ({
  model: null,
  setModel: (model) => set({ model }),

  nodes: [],
  edges: [],
  setGraph: (nodes, edges) => set({ nodes, edges }),

  simResult: null,
  setSimResult: (result) => set({ simResult: result }),
  simRunning: false,
  setSimRunning: (running) => set({ simRunning: running }),

  agentPopulation: [],
  setAgentPopulation: (agents) => set({ agentPopulation: agents }),

  showAgentBridge: true,
  setShowAgentBridge: (show) => set({ showAgentBridge: show }),
  selectedTimestep: 0,
  setSelectedTimestep: (t) => set({ selectedTimestep: t }),

  hasResult: () => get().simResult !== null,
}));
