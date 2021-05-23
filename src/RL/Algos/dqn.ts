import { Agent } from '../Agent';
import { Dynamics, Environment, RepToState, StateToRep } from '../Environments';
import { Space } from '../Spaces';
import seedrandom from 'seedrandom';
import { prng } from '../utils/random';
import * as tf from '@tensorflow/tfjs';

export class DQN<ActionSpace extends Space<Action>, ObservationSpace extends Space<State>, Action, State> extends Agent<
  State,
  Action
> {
  act(observation: State): Action {
    throw new Error('Method not implemented.');
  }
  seed(seed: number) {
    
  }
}
