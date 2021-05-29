import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import nj, { NdArray } from 'numjs';
import { Box, Discrete, Space } from '../../Spaces';
import { Distribution } from '../../utils/Distributions';
import { Normal } from '../../utils/Distributions/normal';


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
  apply(obs: Observation, act: tf.Tensor | null = null) {
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

export class MLPGaussianActor<Observation extends tf.Tensor> extends ActorBase<Observation> {
  public mu_net: tf.LayersModel;
  public log_std: tf.Variable;
  constructor(obs_dim: number, public act_dim: number, hidden_sizes: number[], activation: ActivationIdentifier) {
    super();
    this.log_std = tf.variable(tf.ones([act_dim], 'float32').mul(-0.5), true, 'gaussian-actor-log-std');
    this.mu_net = createMLP(obs_dim, act_dim, hidden_sizes, activation, 'MLP Gaussian Actor');
  }
  _distribution(obs: Observation) {
    const mu = this.mu_net.apply(obs) as tf.Tensor;
    const std = tf.exp(this.log_std);
    return new Normal(mu, std);
  }
  _log_prob_from_distribution(pi: Normal, act: tf.Tensor): tf.Tensor {
    // TODO: check need sum(-1)? torch needs it
    return pi.logProb(act);
  }
}

// TODO:
export class MLPCategoricalActor<Observation extends tf.Tensor> extends ActorBase<Observation> {
  public logits_net: tf.LayersModel;
  constructor(obs_dim: number, act_dim: number, hidden_sizes: number[], activation: ActivationIdentifier) {
    super();
    this.logits_net = createMLP(obs_dim, act_dim, hidden_sizes, activation);
  }
  _distribution(obs: Observation): Distribution {
    throw new Error('Method not implemented.');
  }
  _log_prob_from_distribution(pi: Distribution, act: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    throw new Error('Method not implemented.');
  }
}

export class MLPCritic<Observation extends tf.Tensor> extends Critic<Observation> {
  public v_net: tf.LayersModel;
  constructor(obs_dim: number, hidden_sizes: number[], activation: ActivationIdentifier) {
    super();
    this.v_net = createMLP(obs_dim, 1, hidden_sizes, activation, 'MLP Critic');
  }
  apply(obs: Observation) {
    // TODO check need squeeze?
    return this.v_net.apply(obs) as tf.Tensor;
  }
}

export class MLPActorCritic<Observation extends tf.Tensor> extends ActorCritic<Observation> {
  public pi: Actor<Observation>;
  public v: Critic<Observation>;
  constructor(
    public observationSpace: Space<any>,
    public actionSpace: Space<any>,
    hidden_sizes: number[],
    activation: ActivationIdentifier
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
  step(obs: Observation) {
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
  act(obs: Observation) {
    return this.step(obs).a;
  }
}
