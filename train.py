# _*_ coding: utf-8 _*_
import copy
import os
from typing import NamedTuple, Tuple

import optuna
import tensorflow as tf

import input_data
import mnist


class Settings(NamedTuple):
    learning_rate: int = 0.01
    max_steps: int = 2000
    hidden1: int = 128
    hidden2: int = 32
    batch_size: int = 100
    log_dir: str = '/tmp/tensorflow/mnist/logs/fully_connected_feed'


def placeholder_inputs(batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    images_placeholder = tf.placeholder(
        tf.float32,
        shape=(batch_size, mnist.IMAGE_PIXELS),
    )
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(
        data_set,
        images_pl: tf.Tensor,
        labels_pl: tf.Tensor,
        settings: Settings):
    images_feed, labels_feed = data_set.next_batch(
        settings.batch_size,
        False,
    )
    return {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }


def do_eval(sess: tf.Session,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            settings: Settings) -> float:
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // settings.batch_size
    num_examples = steps_per_epoch * settings.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(
            data_set,
            images_placeholder,
            labels_placeholder,
            settings,
        )
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    # print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
    #       (num_examples, true_count, precision))
    return precision


DATASETS = input_data.read_data_sets(
    '/tmp/tensorflow/mnist/input_data',
    False,  # fake data
)


def run_training(settings: Settings) -> float:
    tf.gfile.MakeDirs(settings.log_dir)

    data_sets = copy.deepcopy(DATASETS)

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            settings.batch_size,
        )

        logits = mnist.inference(
            images_placeholder,
            settings.hidden1,
            settings.hidden2,
        )

        loss = mnist.loss(logits, labels_placeholder)

        train_op = mnist.training(loss, settings.learning_rate)

        eval_correct = mnist.evaluation(logits, labels_placeholder)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(settings.log_dir, sess.graph)

        sess.run(init)

        for step in range(settings.max_steps):

            feed_dict = fill_feed_dict(
                data_sets.train,
                images_placeholder,
                labels_placeholder,
                settings,
            )

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == settings.max_steps:
                checkpoint_file = os.path.join(settings.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                # print('Training Data Eval:')
                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train,
                    settings,
                )
                # Evaluate against the validation set.
                # print('Validation Data Eval:')
                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation,
                    settings,
                )
                # Evaluate against the test set.
                # print('Test Data Eval:')
                acc = do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test,
                    settings,
                )
    return 1 - acc


def objective(trial: optuna.trial.Trial):
    settings = Settings(
        learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        hidden1=trial.suggest_int('hidden1', 50, 200),
        hidden2=trial.suggest_int('hidden2', 1, 100),
        batch_size=trial.suggest_int('batch_size', 1, 100),
    )
    val_err = run_training(settings)
    return val_err


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))

# vim:set fenc=utf-8 ff=unix expandtab sw=4 ts=4:
