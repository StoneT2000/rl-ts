import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import nj, { NdArray } from 'numjs';
import { Box, Discrete, Space } from '../Spaces';
import { Distribution } from '../utils/Distributions';
import { Normal } from '../utils/Distributions/normal';

/** Create a MLP model */
export const createMLP = (
  in_dim: number,
  out_dim: number,
  hidden_sizes: number[],
  activation: ActivationIdentifier,
  name?: string
) => {
  let input = tf.input({ shape: [in_dim] });
  let layer = tf.layers.dense({ units: hidden_sizes[0], activation }).apply(input);
  for (const size of hidden_sizes.slice(1)) {
    layer = tf.layers.dense({ units: size, activation }).apply(layer);
  }
  layer = tf.layers.dense({ units: out_dim, activation: 'linear' }).apply(layer);
  return tf.model({ inputs: input, outputs: layer as SymbolicTensor, name });
};

export abstract class Actor<Observation extends tf.Tensor> {
  abstract _distribution(obs: Observation): Distribution;
  abstract _log_prob_from_distribution(pi: Distribution, act: tf.Tensor): tf.Tensor;
  abstract apply(obs: Observation, act: tf.Tensor): { pi: Distribution; logp_a: tf.Tensor | null };
}
export abstract class Critic<Observation extends tf.Tensor> {
  abstract apply(obs: Observation): tf.Tensor;
}
export abstract class ActorCritic<Observation extends tf.Tensor> {
  abstract pi: Actor<Observation>;
  abstract v: Critic<Observation>;
  abstract step(
    obs: Observation
  ): {
    a: tf.Tensor;
    logp_a: tf.Tensor | null;
    v: tf.Tensor;
  };
  abstract act(obs: Observation): tf.Tensor;
}

export abstract class ActorBase<Observation extends tf.Tensor> extends Actor<Observation> {
  apply(obs: Observation, act: tf.Tensor | null) {
    let pi = this._distribution(obs);
    let logp_a = null;
    if (act !== null) {
      logp_a = this._log_prob_from_distribution(pi, act);
    }
    return {
      pi,
      logp_a,
    };
  }
}

export class MLPGaussianActor extends ActorBase<tf.Tensor> {
  public mu_net: tf.LayersModel;
  public log_std: tf.Variable;
  public mu: tf.Variable;
  constructor(obs_dim: number, public act_dim: number, hidden_sizes: number[], activation: ActivationIdentifier) {
    super();
    this.log_std = tf.variable(tf.ones([act_dim], 'float32').mul(-0.5), true, 'gaussian-actor-log-std');
    this.mu_net = createMLP(obs_dim, act_dim, hidden_sizes, activation, 'MLP Gaussian Actor');
    this.mu = tf.variable(tf.tensor(0));
  }
  _distribution(obs: tf.Tensor) {
    let mu = this.mu_net.apply(obs) as tf.Tensor; // [B, act_dim]
    // let mu = tf.zeros([obs.shape[0], this.act_dim]).add(this.mu);
    let batch_size = mu.shape[0];
    // console.log(this.log_std);
    const std = tf.exp(this.log_std).expandDims(0).tile([batch_size, 1]); // from [act_dim] shaped to [B, act_dim]
    return new Normal(mu, std);
  }
  _log_prob_from_distribution(pi: Normal, act: tf.Tensor): tf.Tensor {
    // TODO: check need sum(-1)? torch needs it
    return pi.logProb(act).sum(-1);
  }
}

// TODO:
export class MLPCategoricalActor extends ActorBase<tf.Tensor> {
  public logits_net: tf.LayersModel;
  constructor(obs_dim: number, act_dim: number, hidden_sizes: number[], activation: ActivationIdentifier) {
    super();
    this.logits_net = createMLP(obs_dim, act_dim, hidden_sizes, activation);
  }
  _distribution(obs: tf.Tensor): Distribution {
    throw new Error('Method not implemented.');
  }
  _log_prob_from_distribution(pi: Distribution, act: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    throw new Error('Method not implemented.');
  }
}

export class MLPCritic extends Critic<tf.Tensor> {
  public v_net: tf.LayersModel;
  constructor(obs_dim: number, hidden_sizes: number[], activation: ActivationIdentifier) {
    super();
    this.v_net = createMLP(obs_dim, 1, hidden_sizes, activation, 'MLP Critic');
  }
  apply(obs: tf.Tensor) {
    // TODO check need squeeze?
    return this.v_net.apply(obs) as tf.Tensor;
  }
}

export class MLPActorCritic extends ActorCritic<tf.Tensor> {
  public pi: Actor<tf.Tensor>;
  public v: Critic<tf.Tensor>;
  constructor(
    public observationSpace: Space<any>,
    public actionSpace: Space<any>,
    hidden_sizes: number[],
    activation: ActivationIdentifier = 'tanh'
  ) {
    super();
    const obs_dim = observationSpace.shape[0];
    const act_dim = actionSpace.shape[0];
    if (actionSpace instanceof Box) {
      this.pi = new MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation);
    } else if (actionSpace instanceof Discrete) {
      this.pi = new MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation);
    } else {
      throw new Error('This action space is not supported');
    }
    this.v = new MLPCritic(obs_dim, hidden_sizes, activation);
  }
  step(obs: tf.Tensor) {
    const pi = this.pi._distribution(obs);
    const a = pi.sample();
    const logp_a = this.pi._log_prob_from_distribution(pi, a);
    const v = this.v.apply(obs);
    return {
      a,
      logp_a,
      v,
    };
  }
  act(obs: tf.Tensor) {
    return this.step(obs).a;
  }
}
