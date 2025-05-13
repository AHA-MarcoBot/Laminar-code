import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from sklearn.metrics.pairwise import cosine_similarity
import GAN_utility as ganModel
import dill
import copy
import random
import csv
from sklearn.manifold import TSNE

def load_data(path, maxlen=None, minlen=0, traces=0, dnn_type=None, openw=False):
    """Load and shape the dataset"""
    npzfile = np.load(path, allow_pickle=True)
    data = npzfile["data"]
    labels = npzfile["labels"]

    npzfile.close()
    return data, labels

VERBOSE = 1
"""预训练模型加载"""
model_save_path = "./model/origin_AWFdata_DF_epoch=30_model.h5"
model = models.load_model(model_save_path)
model.trainable = False

FE=tf.keras.models.Model(model.input,model.get_layer('fc2').output)
Flatten=tf.keras.models.Model(model.input,model.get_layer('flatten').output)

def adjust_WF_data(x = None,perturbation = None, dont_extend = False):
    if dont_extend:
        nonzero_mask = tf.cast(x != 0, tf.float32)
        idx_range = tf.cast(tf.range(tf.shape(x)[-1]), tf.float32)
        last_nonzero_idx = tf.reduce_max(idx_range * nonzero_mask, axis=-1, keepdims=True)
        mask = tf.cast(tf.range(tf.shape(x)[-1], dtype=tf.float32) <= last_nonzero_idx, tf.float32)

    perturbation = tf.expand_dims(perturbation, 2)
    perturbation = perturbation * 1.0

    if dont_extend:
        advData = x + perturbation * tf.sign(x) * mask
    else:
        advData = x + perturbation * tf.sign(x)

    return advData

def adjust_WF_data_patch(x = None,perturbation = None, position=0):

    batch_size = tf.shape(x)[0]
    seq_length = tf.shape(x)[1]
    patch_length = tf.shape(perturbation)[1]

    end = tf.minimum(position + patch_length, seq_length)
    actual_patch_len = end - position
    full_perturbation = tf.zeros_like(x)
    patch = perturbation[:, :actual_patch_len]  # shape: (batch, actual_patch_len)
    patch = tf.expand_dims(patch, axis=-1)      # shape: (batch, actual_patch_len, 1)


    perturbation = tf.expand_dims(perturbation, 2)
    perturbation = perturbation * 1.0
    left = tf.zeros((batch_size, position, 1), dtype=x.dtype)
    right = tf.zeros((batch_size, seq_length - position - actual_patch_len, 1), dtype=x.dtype)
    full_perturbation = tf.concat([left, patch, right], axis=1)  # shape: (batch, 2000, 1)
    advData = x + full_perturbation * tf.sign(x)

    return advData

def get_class_samples(X, Y, C):
    y = np.argmax(Y, axis=1)
    ind = np.where(y == C)
    return X[ind], Y[ind]

generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def overHead_loss(X_ori, X_adv, overHead_thresh=0.1):
    overHead = tf.reduce_sum(tf.abs(X_adv)-tf.abs(X_ori))/tf.reduce_sum(tf.abs(X_ori))
    return tf.maximum(0.0, overHead-overHead_thresh)

def get_other_class(pred, targetLabel):
    mean = tf.reduce_mean(pred, axis=0)
    masked_means = tf.tensor_scatter_nd_update(mean, [[targetLabel]], [float('-inf')])
    max_class_idx = tf.argmax(masked_means)
    return max_class_idx

def draw_Columnar(label,a,b,c):
    x = np.arange(len(a))

    a_abs = np.abs(a)
    b_delta = np.abs(b) - a_abs
    c_delta = np.abs(c) - np.abs(b)

    plt.figure()
    plt.bar(x, a_abs, color='lightgray', label='|a|')
    plt.bar(x, b_delta, bottom=a_abs, color='skyblue', label='|b| - |a|')
    plt.bar(x, c_delta, bottom=a_abs + b_delta, color='orange', label='|c| - |b|')
    plt.title("Stacked Bar Showing Magnitude Increase from |a| → |b| → |c|")
    plt.xlabel("Index")
    plt.ylabel("Absolute Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plt/Columnar/{label}.png")

batch_size = 64
g_iteration = 200
data_length = 2000
num_classes = 200
patch_length = 10

classNum = []
for i in range(0, num_classes):
    classNum.append(i)

data, labels = load_data("./dataset/Burst_Closed World/burst_tor_200w_2500tr_test.npz")
data = data.reshape((data.shape[0], data.shape[2], 1))
data = data.astype("float32")

pairs = []
perturbation_rate = []
morphing_rate = []
total_overhead = []

seed = [22, 33]
random_row = tf.random.stateless_uniform(shape=(1, data_length), seed=seed, minval=1, maxval=200, dtype=tf.int32)
signs = tf.pow(-1, tf.range(data_length, dtype=tf.int32) + 1)
random_noise = random_row * tf.expand_dims(signs, axis=0)
mask_noise_X = tf.tile(random_noise, [batch_size, 1])
print("mask_noise_X shape:", mask_noise_X.shape)

a = tf.random.uniform((), minval=0, maxval=200-2*patch_length, dtype=tf.int32, seed=seed[0])
b = tf.random.uniform((), minval=a+10, maxval=200-patch_length, dtype=tf.int32, seed=seed[1])
print("Inject location: ", a, b)

start = 0
for label in range(start, start+50):
    train_data_X, train_data_Y = get_class_samples(data, labels, label)

    test_data_X = train_data_X[1200:, :, :]
    test_data_Y = train_data_Y[1200:, :]

    train_data_X = train_data_X[:1200, :, :]
    train_data_Y = train_data_Y[:1200, :]

    print(train_data_X.shape, train_data_Y.shape)

    forward_generator = ganModel.generator_model_5()
    forward_discriminator = ganModel.discriminator_model_5()
    targetLabel = 199
    patch_generator_1 = ganModel.patch_model_10()
    patch_generator_2 = ganModel.patch_model_10()

    pairs.append((label, targetLabel))

    targe_data_X, targe_data_Y = get_class_samples(data, labels, targetLabel)
    targe_data_X = targe_data_X[:1200, :, :]
    targe_data_Y = targe_data_Y[:1200, :]

    gen_loss = []
    total_loss = []
    diff_loss = []
    head_one = []
    kl_loss = tf.keras.losses.KLDivergence()


    for iter in range(g_iteration):
        with tf.GradientTape() as G1_tape, tf.GradientTape() as G4_tape:
            indices = np.random.randint(train_data_X.shape[0], size=batch_size)
            x_train_batch = train_data_X[indices]
            y_train_batch = train_data_Y[indices]

            x_mask_batch = tf.expand_dims(mask_noise_X, 1)

            random_noise_one = np.random.normal(size=[batch_size, data_length])
            adv_distribution_a = patch_generator_1(random_noise_one, training=True)

            generated_one = adjust_WF_data_patch(x_train_batch, adv_distribution_a, position=a)

            random_noise_one = np.random.normal(size=[batch_size, data_length])
            adv_distribution_b = patch_generator_2(random_noise_one, training=True)
            generated_one = adjust_WF_data_patch(generated_one, adv_distribution_b, position=b)

            adjusted_generated = tf.expand_dims(generated_one, 1)

            pre_one = model(adjusted_generated)

            gen_diff_loss = -tf.math.log(1 - tf.reduce_mean(tf.maximum(pre_one[:, label] - 1e-8, 0)))

            origin_target_loss = 0.01 * kl_loss(FE(adjusted_generated), FE(x_mask_batch))

            head_loss = overHead_loss(x_train_batch, generated_one)
 
            loss = gen_diff_loss + origin_target_loss + tf.abs(0.2 - head_loss)
            total_loss.append(loss.numpy())
            gen_loss.append(origin_target_loss.numpy())
            diff_loss.append(gen_diff_loss.numpy())
            head_one.append(head_loss.numpy())

            if iter % 50 == 0:
                print("##############gen_diff_loss", gen_diff_loss.numpy())
                print("##############origin_target_loss", origin_target_loss.numpy())
        
        gradient_gen_1 = G1_tape.gradient(loss, patch_generator_1.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradient_gen_1, patch_generator_1.trainable_variables))

        gradient_gen_2 = G4_tape.gradient(loss, patch_generator_2.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradient_gen_2, patch_generator_2.trainable_variables))
    print("overhead: ", head_loss.numpy())

    plt.figure()
    plt.plot(head_one, label="overhead")
    plt.plot(total_loss, label="total_loss")
    plt.plot(diff_loss, label="origin_class_loss")
    plt.plot(gen_loss, label="cosine_similarity_loss")
    # plt.plot(dis_loss, label="dis_loss")
    plt.title(" Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./plt/mask/" + str(label) + "_loss.png")
    plt.close()
    head_one = []
    total_loss = []
    diff_loss = []
    gen_loss = []
    
    random_noise_genmask_train = np.random.normal(size=[1200, data_length])  # 交给生成器的随机输入
    random_noise_genmask_test = np.random.normal(size=[300, data_length])
    mask_noise_train_1 = patch_generator_1(random_noise_genmask_train, training=False)
    mask_noise_test_1 = patch_generator_1(random_noise_genmask_test, training=False)
    mask_noise_train_2 = patch_generator_2(random_noise_genmask_train, training=False)
    mask_noise_test_2 = patch_generator_2(random_noise_genmask_test, training=False)

    mask_noise_train_1 = tf.where(tf.math.is_nan(mask_noise_train_1), tf.zeros_like(mask_noise_train_1), mask_noise_train_1)
    mask_noise_test_1 = tf.where(tf.math.is_nan(mask_noise_test_1), tf.zeros_like(mask_noise_test_1), mask_noise_test_1)
    mask_noise_train_2 = tf.where(tf.math.is_nan(mask_noise_train_2), tf.zeros_like(mask_noise_train_2), mask_noise_train_2)
    mask_noise_test_2 = tf.where(tf.math.is_nan(mask_noise_test_2), tf.zeros_like(mask_noise_test_2), mask_noise_test_2)

    adjusted_masked_x_train = adjust_WF_data_patch(train_data_X, mask_noise_train_1, position=a)
    adjusted_masked_x_test = adjust_WF_data_patch(test_data_X, mask_noise_test_1, position=a)
    adjusted_masked_x_train = adjust_WF_data_patch(adjusted_masked_x_train, mask_noise_train_2, position=b)
    adjusted_masked_x_test = adjust_WF_data_patch(adjusted_masked_x_test, mask_noise_test_2, position=b)

    head_one = []
    total_loss = []
    logit_loss = []
    gen_loss = []

    disc_total = []
    disc_source = []
    disc_target = []
    disc_fake = []

    bce_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    real_labels = tf.constant(1, shape=(batch_size,), dtype=tf.float32)
    fake_labels = tf.constant(0, shape=(batch_size,), dtype=tf.float32)
    target_labels = np.zeros((batch_size, num_classes))
    target_labels[:, targetLabel] = 1
    
    for iter in range(g_iteration + 300):
        with tf.GradientTape() as G2_tape, tf.GradientTape() as G3_tape:
            indices = np.random.randint(targe_data_X.shape[0], size=batch_size)
            x_targe_batch = targe_data_X[indices]
            x_targe_batch = tf.expand_dims(x_targe_batch, 1)
            y_targe_batch = targe_data_Y[indices]
            
            indices = np.random.randint(adjusted_masked_x_train.shape[0], size=batch_size)
            indices = tf.convert_to_tensor(indices, dtype=tf.int32)
            x_train_batch = tf.gather(adjusted_masked_x_train, indices)

            random_noise_two = np.random.normal(size=[batch_size, data_length])
            adv_distribution = forward_generator(random_noise_two, training=True)

            generated_two = adjust_WF_data(x_train_batch, adv_distribution)
            head_loss = overHead_loss(x_train_batch, generated_two)

            adjusted_generated = tf.expand_dims(generated_two, 1)
            pre_two = model(adjusted_generated)
            disc_pre_two = forward_discriminator(tf.squeeze(adjusted_generated), training=False)
            
            other_class = get_other_class(pre_two, targetLabel)
            gen_logit_loss= -tf.math.log(tf.reduce_mean(tf.maximum(pre_two[:, targetLabel], 1e-8))) -tf.math.log(1 - tf.reduce_mean(tf.maximum(pre_two[:, other_class] - 1e-8, 0)))
            origin_target_loss = bce_loss(real_labels, disc_pre_two) / 16.0

            loss = gen_logit_loss + origin_target_loss + 0.7 * head_loss

            total_loss.append(loss.numpy())
            gen_loss.append(origin_target_loss.numpy())
            logit_loss.append(gen_logit_loss.numpy())
            head_one.append(head_loss.numpy())

            real_target_output = forward_discriminator(tf.squeeze(x_targe_batch), training=True)
            real_source_output = forward_discriminator(tf.squeeze(x_train_batch), training=True)
            fake_output = forward_discriminator(tf.squeeze(adjusted_generated), training=True)

            disc_loss = 2.0 * bce_loss(real_labels, real_target_output) + bce_loss(fake_labels, fake_output) + bce_loss(fake_labels, real_source_output)
            disc_total.append(disc_loss.numpy())
            disc_target.append(bce_loss(real_labels, real_target_output).numpy())
            disc_source.append(bce_loss(fake_labels, real_source_output).numpy())
            disc_fake.append(bce_loss(fake_labels, fake_output).numpy())
            
            if iter % 50 == 0:
                print("##############subtitute_loss", gen_logit_loss.numpy())
                print("##############disc_loss", origin_target_loss.numpy())
                if iter == 0:
                    G2_early_stop = loss
                elif tf.math.abs(G2_early_stop - loss) < 0.02:
                    print("G2 earlt stop.")
                    break
                else:
                    G2_early_stop = loss

        gradient_gen = G2_tape.gradient(loss, forward_generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradient_gen, forward_generator.trainable_variables))

        gradient_disc = G3_tape.gradient(disc_loss, forward_discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradient_disc, forward_discriminator.trainable_variables))

    print("overhead:", head_loss.numpy())
    forward_generator.save_weights("./generator/forward/" + str(label) + ".h5")
    plt.figure()
    plt.plot(head_one, label="overhead")
    plt.plot(total_loss, label="total_loss")
    plt.plot(logit_loss, label="discriminator_loss")
    plt.plot(gen_loss, label="consine_similarity_loss")

    plt.title(" Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./plt/forward/" + str(label) + "_loss.png")
    plt.close()
    head_one = []
    total_loss = []
    logit_loss = []
    gen_loss = []

    plt.figure()
    plt.plot(disc_total, label="total_loss")
    plt.plot(disc_source, label="source_loss")
    plt.plot(disc_target, label="target_loss")
    plt.plot(disc_fake, label="fake_loss")
    plt.title("Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./plt/forward/" + '_disc_' + str(label) + "_loss.png")
    plt.close()
    disc_total = []
    disc_source = []
    disc_target = []
    disc_fake = []

    add_forward = ganModel.generator_model_5()
    add_forward.build(input_shape=(None, data_length))
    add_forward.load_weights("./generator/forward/" + str(label) + ".h5")
    add_forward.trainable = False

    random_noise_genforward_train = np.random.normal(size=[1200, data_length])
    random_noise_genforward_test = np.random.normal(size=[300, data_length])
    forward_noise_train = add_forward(random_noise_genforward_train, training=False)
    forward_noise_test = add_forward(random_noise_genforward_test, training=False)

    adjusted_forwarded_x_train = adjust_WF_data(adjusted_masked_x_train, forward_noise_train)
    adjusted_forwarded_x_test = adjust_WF_data(adjusted_masked_x_test, forward_noise_test)

    if label == start:
        forward_X_train = adjusted_forwarded_x_train
        forward_Y_train = train_data_Y
        forward_X_test = adjusted_forwarded_x_test
        forward_Y_test = test_data_Y
    else:
        forward_X_train = np.concatenate([forward_X_train, adjusted_forwarded_x_train], axis=0)
        forward_Y_train = np.concatenate([forward_Y_train, train_data_Y], axis=0)
        forward_X_test = np.concatenate([forward_X_test, adjusted_forwarded_x_test], axis=0)
        forward_Y_test = np.concatenate([forward_Y_test, test_data_Y], axis=0)
    
    total_overhead.append(overHead_loss(train_data_X, adjusted_forwarded_x_train).numpy())

    data_before_X, data_before_Y = get_class_samples(data, labels, label)
    data_before_X = tf.expand_dims(data_before_X, 1)
    data_after_X = np.concatenate([adjusted_forwarded_x_train, adjusted_forwarded_x_test], axis=0)
    data_after_X = tf.expand_dims(data_after_X, 1)
    _, data_after_Y = get_class_samples(data, labels, targetLabel)
    before = model.evaluate(data_before_X, data_before_Y)
    after = model.evaluate(data_after_X, data_before_Y)
    pert = (before[1] - after[1]) / before[1] 
    perturbation_rate.append(pert)
    morp = model.evaluate(data_after_X, data_after_Y)
    morphing_rate.append(morp[1])

    draw_Columnar(label, tf.squeeze(train_data_X[0]), tf.squeeze(adjusted_masked_x_train[0]), tf.squeeze(adjusted_forwarded_x_train[0]))

    
plt.figure()
plt.plot(perturbation_rate, label="perturbation_rate", linestyle='-', color='aquamarine', marker='s', markerfacecolor='aquamarine', markersize=3, linewidth=1)
plt.plot(morphing_rate, label="morphing_rate", linestyle='-', color='deepskyblue', marker='s', markerfacecolor='deepskyblue', markersize=3, linewidth=1)
# plt.plot(dis_loss, label="dis_loss")
plt.title("White_box_DF")
plt.xlabel("source_label")
plt.ylabel("rate(%)")
plt.legend()
plt.savefig("rate.png")
plt.close()


print(forward_X_train.shape, forward_Y_train.shape)
print(forward_X_test.shape, forward_Y_test.shape)
forward_X_train = np.round(forward_X_train)
forward_X_test = np.round(forward_X_test)

myind_train = list(range(forward_X_train.shape[0]))
random.shuffle(myind_train)
forward_X_train = forward_X_train[myind_train]
forward_Y_train = forward_Y_train[myind_train]

myind_test = list(range(forward_X_test.shape[0]))
random.shuffle(myind_test)
forward_X_test = forward_X_test[myind_test]
forward_Y_test = forward_Y_test[myind_test]

print("total overehead: ", np.nanmean(total_overhead))

dill.dump((forward_X_train, forward_Y_train), open("./advdata/train.dill", "wb"))
dill.dump((forward_X_test, forward_Y_test), open("./advdata/test.dill", "wb"))

with open("./pairs/pairs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Number1", "Number2"])
    writer.writerows(pairs)