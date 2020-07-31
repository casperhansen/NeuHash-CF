import argparse
import tensorflow as tf
import pickle
import glob
from nn_helpers import generator
from model import Model
import numpy as np
import time
from helpers import ndcg_score
from scipy.io import loadmat
import datetime
from bit_inspecter import bit_histogram
import multiprocessing as mp
from scipy.io import loadmat
import time

def fprint(output_file, text):
    with open(output_file, "a") as myfile:
        myfile.write(str(text) + "\n")

def onepass_test(preloaded_testsamples, sess, eval_list, some_handle, num_samples, eval_batchsize,
                 handle, anneal_val, anneal_val_vae, batch_placeholder, is_training, force_selfmask, args, num_items, num_users,
                 full_matrix_ratings, test_samples):

    valcounter = 0

    item_matrix = np.zeros((num_items, args["bits"]))
    user_matrix = np.zeros((num_users, args["bits"]))

    total = num_samples
    valdone = False

    all_item_ids = []
    all_user_ids = []
    while not valdone:
        lossval, hamdist, userval, item_ratingval, itemsample, item_emb, user_emb, \
            lossval_recon, lossval_vae = sess.run(eval_list, feed_dict={handle: some_handle, is_training: False,
                                                                        anneal_val: 0,
                                                                        anneal_val_vae: 0,
                                                                        batch_placeholder: min(total, eval_batchsize)})

        valcounter += 1
        total -= len(userval)

        #print(hamdist)
        if total <= 0:
            valdone = True

        item_emb = item_emb.tolist()
        user_emb = user_emb.tolist()

        for kk in range(len(userval)):
            user = userval[kk]
            item = itemsample[kk]

            item_matrix[item] = item_emb[kk]
            user_matrix[user] = user_emb[kk]

            all_item_ids.append(item)
            all_user_ids.append(user)

    assert(total == 0)
    assert(len(set(all_item_ids)) == num_items)
    assert(len(set(all_user_ids)) == num_users)

    #print("test", np.mean(bit_histogram(user_matrix, 1)), np.mean(bit_histogram(item_matrix, 1)))
    #fprint(args["ofile"],"test" + " ".join([str(v) for v in [np.mean(bit_histogram(user_matrix, 1)), np.mean(bit_histogram(item_matrix, 1))]]))


    #ndcgs, mse = eval_hashing(full_matrix_ratings, test_samples[0], item_matrix, user_matrix, num_users, num_items, args)

    ndcgs, mse = eval_hashing_fast(preloaded_testsamples, item_matrix, user_matrix, args["bits"])

    #print("####", np.mean(ndcgs, 0))
    return np.mean(ndcgs,0), mse, item_matrix, user_matrix

def eval_hashing(full_matrix_ratings, test_samples, item_matrix, user_matrix, num_users, num_items, args):
    inps = []
    mses = []
    #start = time.time()
    for user in (range(num_users)):
        user_emb = user_matrix[user]
        items = test_samples[user][0]#[0]
        item_ids = items



        items = item_matrix[item_ids]# np.array([item_matrix[item] for item in items])
        user_emb_01 = (user_emb+1)/2

        if args["force_selfmask"]:
            ham_dists = np.array([np.sum(user_emb * ( 2*((item+1)/2 * user_emb_01) - 1)) for item in items])
        else:
            ham_dists = np.array([np.sum(user_emb * item) for item in items])
            
        items_gt = np.squeeze(np.array(full_matrix_ratings[user, item_ids].todense()), 0) #, axis=-1) #[full_matrix_ratings[user,iid] for iid in item_ids] #
        inps.append([items_gt, args["bits"]-ham_dists])

        tmp_gt = 2*args["bits"] * items_gt/5.0 - args["bits"]
        mse = (tmp_gt - ham_dists)**2 #/ args["batchsize"]
        mses += mse.tolist()

    ndcgs = [ndcg_score(inp[0], inp[1]) for inp in inps]
    #print("eval duration", time.time() - start )
    return ndcgs, np.mean(mses)

def eval_hashing_fast(test_samples, item_matrix, user_matrix, numbits):
    inps = []
    mses = []
    # start = time.time()
    num_users = len(user_matrix)
    #tt = time.time()
    totalnumitems = 0
    for user in (range(num_users)):
        user_emb = user_matrix[user]
        if len(test_samples[user]) == 2 and -1 == test_samples[user][0]:
            continue
        #print(test_samples[user])
        items = np.array([elm[0] for elm in test_samples[user]])# [0]
        items_gt = np.array([elm[1] for elm in test_samples[user]])

        item_ids = items
        totalnumitems += len(item_ids)
        #print(item_ids)

        #print(item_matrix.shape)

        items = item_matrix[item_ids]  # np.array([item_matrix[item] for item in items])
        user_emb_01 = (user_emb + 1) / 2

        ham_dists = np.array([np.sum(user_emb * item) for item in items])

        #items_gt = np.squeeze(np.array(full_matrix_ratings[user, item_ids].todense()),
        #                      0)  # , axis=-1) #[full_matrix_ratings[user,iid] for iid in item_ids] #
        inps.append([items_gt, numbits - ham_dists])

        tmp_gt = 2 * numbits * items_gt / 5.0 - numbits
        mse = (tmp_gt - ham_dists) ** 2  # / args["batchsize"]
        mses += mse.tolist()

    #print(time.time() - tt )
    #tt = time.time()
    ndcgs = [ndcg_score(inp[0], inp[1]) for inp in inps]
    #print("last",time.time() - tt)

    # print("eval duration", time.time() - start )
    return ndcgs, np.mean(mses) # ndcgs, np.mean(mses), totalnumitems

def onepass(sess, eval_list, some_handle, num_samples, eval_batchsize, handle, anneal_val, anneal_val_vae, batch_placeholder, is_training, force_selfmask, args):
    losses_val = []
    losses_val_recon = []
    losses_val_vae = []

    losses_val_eq = []
    losses_val_uneq = []

    valcounter = 0
    val_user_items = {}
    total = num_samples
    valdone = False

    user_vectors = []
    item_vectors = []

    while not valdone:
        lossval, hamdist, userval, item_ratingval, itemsample, item_emb, user_emb, \
            lossval_recon, lossval_vae = sess.run(eval_list, feed_dict={handle: some_handle, is_training: False,
                                                                        anneal_val: 0,
                                                                        anneal_val_vae: 0,
                                                                        batch_placeholder: min(total, eval_batchsize)})

        losses_val.append(lossval)

        losses_val_recon.append(lossval_recon)
        losses_val_vae.append(lossval_vae)
        #losses_val_uneq.append(lossval_uneq)
        #losses_val_eq.append(lossval_eq)

        valcounter += 1
        total -= len(userval)
        user_vectors += user_emb.tolist()
        item_vectors += item_emb.tolist()

        #print(hamdist)
        if total <= 0:
            valdone = True

        for kk in range(len(userval)):
            user = userval[kk]
            item_rating = item_ratingval[kk]
            #user_item_score = hamdist[kk]

            #print(user_item_score, np.sum(user_emb[kk] * item_emb[kk]))
            #if force_selfmask:
            #    user_msk = user > 0
            #    item_emb = item_emb * user_msk
            #    item_emb[item_emb < 0.5] = -1
            #    #print(user_emb[kk], item_emb[kk])

            user_item_score = -np.sum(user_emb[kk] * item_emb[kk])

            if user not in val_user_items:
                val_user_items[user] = [[], []]

            val_user_items[user][0].append(int(user_item_score))
            val_user_items[user][1].append(int(item_rating))

    assert(total == 0)
    #ndcgs = []
    t = 0

    #val_user_items_from_to = []
    #splits = 4
    #NNN = len(val_user_items)
    #for i in range(splits):
    #    fromval = int(NNN * i / splits)
    #    toval = int(NNN * (i+1) / splits)
    #    val_user_items_from_to.append([fromval, toval])

    inps = []
    for user in val_user_items:
        t += len(val_user_items[user][1])
        #ndcg_val = ndcg_score(val_user_items[user][1], val_user_items[user][0], k=10)
        inps.append([val_user_items[user][1], val_user_items[user][0]])
        #print(val_user_items[user][1], val_user_items[user][0], ndcg_val)
        #ndcgs.append(ndcg_val)
    res = pool.starmap_async(ndcg_score, inps)
    ndcgs = res.get()

    if not args["realvalued"]:
        user_vectors = np.array(user_vectors).astype(int)
        item_vectors = np.array(item_vectors).astype(int)
    else:
        user_vectors = np.array(user_vectors)
        item_vectors = np.array(item_vectors)

    #print(user_vectors[202], item_vectors[202])
    print("val", np.mean(bit_histogram(user_vectors, 1)), np.mean(bit_histogram(item_vectors, 1)))
    print(user_vectors[10][:5], item_vectors[10][:5])
    print(np.unique(user_vectors.astype(int)[:]), np.unique(item_vectors.astype(int)[:]), len(item_vectors[0]), len(user_vectors[0]) )
    fprint(args["ofile"],"val" + " ".join([str(v) for v in [np.mean(bit_histogram(user_vectors, 1)), np.mean(bit_histogram(item_vectors, 1))]]))
    #diffs = (item_vectors[:, :32] != item_vectors[:, 32:])
    #diffs_notUserZero = diffs[user_vectors[:, :32] > 0]
    #print(np.mean(diffs_notUserZero))

    #print("####", sum([len(val_user_items[user][0]) for user in val_user_items]))

    return np.mean(losses_val), np.mean(ndcgs, 0), np.mean(losses_val_recon), \
           np.mean(losses_val_uneq), np.mean(losses_val_eq), len(ndcgs), ndcgs, np.mean(losses_val_vae)

def main():
    parser = argparse.ArgumentParser()
    # Tune/change the follow values for your experiments
    parser.add_argument("--batchsize", default=2000, type=int) # this can be kept fixed.
    parser.add_argument("--bits", default=32, type=int) # 16-64 evaluated
    parser.add_argument("--lr", default=0.0005, type=float) 
    parser.add_argument("--vae_units", default=1000, type=int) 
    parser.add_argument("--vae_layers", default=2, type=int) 
    parser.add_argument("--dataset", default="amacold", type=str) # dataset type
    parser.add_argument("--vae_weight", default=0.001, type=float) # \alpha in eq. 22



    parser.add_argument("--mul", default=6, type=float) # 6 is fine. used to control how often to evaluate. Higher value means less often.
    parser.add_argument("--anneal_val", default=1.0, type=float)
    parser.add_argument("--decay_rate", default=1.0, type=float)
    parser.add_argument("--ofile", default="../output.txt", type=str)

    parser.add_argument("--deterministic_eval", default=1, type=int)
    parser.add_argument("--deterministic_train", default=0, type=int)

    parser.add_argument("--optimize_selfmask", default=0, type=int)
    parser.add_argument("--usermask_nograd", default=0, type=int)
    parser.add_argument("--KLweight", default=0.00, type=float)

    parser.add_argument("--force_selfmask", default=0, type=int)
    parser.add_argument("--save_vectors", default=0, type=int)
    parser.add_argument("--realvalued", default=0, type=int)

    parser.add_argument("--annealing_min", default=0.0, type=float)
    parser.add_argument("--annealing_max", default=1.0, type=float)
    parser.add_argument("--annealing_decrement", default=0.000001, type=float)

    parser.add_argument("--item_emb_type", default=1, type=int) # 0=no item features, 1 = only item features, 2 = combined by multiplication

    parser.add_argument("--loss_type", default="normal", type=str)
    parser.add_argument("--loss_alpha", default=3.0, type=float)

    eval_batchsize = 2000
    args = parser.parse_args()

    savename = "results/" + "_".join([str(v) for v in [args.dataset, args.item_emb_type, args.bits, args.batchsize, args.lr, args.anneal_val, args.deterministic_eval,
                                          args.deterministic_train, args.KLweight, args.vae_units, args.vae_layers, args.vae_weight, args.loss_type, args.loss_alpha]]) + "_res.pkl"

    args.realvalued = args.realvalued > 0.5
    args.deterministic_eval = args.deterministic_eval > 0.5
    args.usermask_nograd = args.usermask_nograd > 0.5
    args.deterministic_train = args.deterministic_train > 0.5
    args.optimize_selfmask = args.optimize_selfmask > 0.5
    args.force_selfmask = args.force_selfmask > 0.5
    args.save_vectors = args.save_vectors > 0.5

    args = vars(args)
    print(args)
    fprint(args["ofile"], args)

    basepath = "../data/"+args["dataset"]+"/tfrecord/"
    dicfile = basepath + "dict.pkl"
    dicfile = pickle.load(open(dicfile, "rb"))
    num_users, num_items = dicfile[0], dicfile[1]

    args["num_users"] = num_users
    args["num_items"] = num_items

    trainfiles = glob.glob(basepath + "*train_*tfrecord")
    valfiles = glob.glob(basepath + "*val_*tfrecord")
    testfiles = glob.glob(basepath + "*test_*tfrecord")

    if args["dataset"].lower() == "yelp":
        train_samples = 531990
        val_samples = 108065
        test_samples = 47413
        max_rating = 5.0
    elif args["dataset"].lower() == "amazon":
        train_samples = 810049
        val_samples = 161688
        test_samples = 73857
        max_rating = 5.0

    elif args["dataset"].lower() == "yelcold": # 50p
        train_samples = 550708
        val_samples = 96024
        test_samples = 47413
        max_rating = 5.0
    elif args["dataset"].lower() == "yelcold_10p":
        train_samples = 130456
        val_samples = 96024
        test_samples = 47413
        max_rating = 5.0
    elif args["dataset"].lower() == "yelcold_20p":
        train_samples = 259672
        val_samples = 96024
        test_samples = 47413
        max_rating = 5.0
    elif args["dataset"].lower() == "yelcold_30p":
        train_samples = 388761
        val_samples = 96024
        test_samples = 47413
        max_rating = 5.0
    elif args["dataset"].lower() == "yelcold_40p":
        train_samples = 517552
        val_samples = 96024
        test_samples = 47413
        max_rating = 5.0

    elif args["dataset"].lower() == "amacold": # 50p
        train_samples = 831866
        val_samples = 148062
        test_samples = 73857
        max_rating = 5.0
    elif args["dataset"].lower() == "amacold_10p":
        train_samples = 196174
        val_samples = 148062
        test_samples = 73857
        max_rating = 5.0
    elif args["dataset"].lower() == "amacold_20p":
        train_samples = 391722
        val_samples = 148062
        test_samples = 73857
        max_rating = 5.0
    elif args["dataset"].lower() == "amacold_30p":
        train_samples = 587182
        val_samples = 148062
        test_samples = 73857
        max_rating = 5.0
    elif args["dataset"].lower() == "amacold_40p":
        train_samples = 782075
        val_samples = 148062
        test_samples = 73857
        max_rating = 5.0

    else:
        #pass
        exit(-1)

    #print(args["dataset"])
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(trainfiles[0])))
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(valfiles[0])))
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(testfiles[0])))
    #exit() # ...

    preloaded_testsamples = pickle.load(open(args["dataset"] + "_testdata.pkl","rb"))

    total_samples = train_samples + val_samples + test_samples
    print(args["dataset"], total_samples, " & ", num_items, " & ", num_users, " & ", str(total_samples/(num_items * num_users)*100) + "\%")

    datamatlab = loadmat('../data/' + args["dataset"] + '/ratings_contentaware_full.mat')
    item_content_matrix = datamatlab["item_features"].todense()

    tf.reset_default_graph()
    with tf.Session() as sess:

        handle = tf.placeholder(tf.string, shape=[], name="handle_iterator")
        training_handle, train_iter, gen_iter = generator(sess, handle, args["batchsize"], trainfiles, 0)
        val_handle, val_iter, _ = generator(sess, handle, eval_batchsize, valfiles, 1)
        test_handle, test_iter, _ = generator(sess, handle, eval_batchsize, testfiles, 1)

        sample = gen_iter.get_next()
        user_sample = sample[0]
        item_sample = sample[1]
        item_rating = sample[4]

        is_training = tf.placeholder(tf.bool, name="is_training")
        anneal_val = tf.placeholder(tf.float32, name="anneal_val", shape=())
        anneal_val_vae = tf.placeholder(tf.float32, name="anneal_val_vae", shape=())

        batch_placeholder = tf.placeholder(tf.int32, name="batch_placeholder")

        model = Model(sample, args)

        item_emb_matrix, item_emb_ph, item_emb_init = model._make_embedding(num_items, args["bits"], "item_embedding")
        content_matrix, content_emb_ph, content_emb_init = model._make_embedding(num_items, 8000, "content_embedding", trainable=False)
        #content_matrix = model.convert_sparse_matrix_to_sparse_tensor(item_content_matrix)
        user_emb_matrix, _, _ = model._make_embedding(num_users, args["bits"], "user_embedding1")

        word_embedding_matrix, _, _ = model._make_embedding(8000, 300, "word_embedding")
        importance_embedding_matrix = model.make_importance_embedding(8000)

        loss, loss_no_anneal, scores, item_embedding, user_embedding, \
            reconloss, loss_vae = model.make_network(word_embedding_matrix, importance_embedding_matrix, content_matrix, item_emb_matrix, user_emb_matrix,
                                           is_training, args, max_rating, anneal_val, anneal_val_vae, batch_placeholder)

        step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(args["lr"],
                                        step,
                                        100000,#int(train_samples / vars_dict["batch_size"]),
                                        args["decay_rate"],
                                        staircase=True, name="lr")

        optimizer = tf.train.AdamOptimizer(learning_rate=args["lr"], name="Adam")
        #optimizer = tf.train.RMSPropOptimizer(lr, name="RMS")
        #optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, name="sgd_momentum")
        train_step = optimizer.minimize(loss)#, global_step=step)

        init = tf.global_variables_initializer()
        sess.run(init)

        #sess.run(user_index_embedding_init, feed_dict={user_index_embedding_ph: user_index_embedding})

        sess.run(content_emb_init, feed_dict={content_emb_ph: item_content_matrix})

        sess.run(train_iter.initializer)

        eval_list = [loss, scores, user_sample, item_rating, item_sample, item_embedding, user_embedding, reconloss, loss_vae]
        counter = 0
        losses_train = []
        losses_train_no_anneal = []
        times = []
        anneal = args["anneal_val"]
        anneal_vae = args["annealing_max"]


        best_val_ndcg = 0
        best_val_loss = np.inf
        patience = 5

        patience_counter = 0
        running = True
        print("starting training")
        all_val_ndcg = []
        while running:

            start = time.time()

            lossval, loss_no_anneal_val, hamdist, _= sess.run([loss, loss_no_anneal, scores, train_step], feed_dict={handle: training_handle, is_training: True,
                                                                                  anneal_val: anneal,
                                                                                  anneal_val_vae: anneal_vae,
                                                                                  batch_placeholder: args["batchsize"]})
            #print(np.mean((iv + sv)**2), lossval)
            #print(np.mean(nzb/args["bits"]), iv[:5], -hamdist[:5], ir[:5], "---iv", np.min(iv), np.max(iv), "--ham:", np.min(-hamdist), np.max(-hamdist) )
            #print(uv[0], iv[0])
            times.append(time.time() - start)
            losses_train.append(lossval)
            losses_train_no_anneal.append(loss_no_anneal_val)
            counter += 1

            anneal_vae = max(anneal_vae-args["annealing_decrement"], args["annealing_min"])

            anneal = anneal * 0.9999
            if counter % int(1500*args["mul"]) == 0:
                print("train", np.mean(losses_train), np.mean(losses_train_no_anneal), counter * args["batchsize"] / train_samples, np.mean(times), anneal)
                fprint(args["ofile"], " ".join([str(v) for v in ["train", np.mean(losses_train), np.mean(losses_train_no_anneal), counter * args["batchsize"] / train_samples, np.mean(times), anneal]]) )
                losses_train = []
                times = []
                losses_train_no_anneal = []

                sess.run(val_iter.initializer)
                losses_val, val_ndcg, losses_val_recon, losses_val_uneq, losses_val_eq, NN, allndcgs, losses_val_vae = onepass(sess, eval_list, val_handle,
                                     val_samples, eval_batchsize,
                                     handle, anneal_val, anneal_val_vae, batch_placeholder, is_training, args["force_selfmask"], args)
                print("val\t\t", val_ndcg, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN, losses_val_vae)
                save_val_ndcg = val_ndcg
                fprint(args["ofile"], " ".join([str(v) for v in ["val\t\t", val_ndcg, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN]]) )

                all_val_ndcg.append(best_val_ndcg)
                if val_ndcg[-1] > best_val_ndcg:# or best_val_loss > losses_val:
                    best_val_ndcg = val_ndcg[-1]
                    best_val_loss = losses_val
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter == 0:
                    sess.run(test_iter.initializer)
                    test_ndcgs, test_mse, user_matrix_vals, item_matrix_vals = onepass_test(preloaded_testsamples,sess, eval_list, test_handle,
                                 test_samples, eval_batchsize,
                                 handle, anneal_val, anneal_val_vae, batch_placeholder,
                                 is_training, args["force_selfmask"], args,
                                 num_items, num_users,
                                 datamatlab["full_matrix"], datamatlab["test_rated_total"])
                    print("test\t\t\t\t",  test_ndcgs, test_mse)
                    fprint(args["ofile"], " ".join([str(v) for v in ["test\t\t\t\t", test_ndcgs, test_mse]]))

                    imp_emb_values = sess.run([importance_embedding_matrix], feed_dict={handle: training_handle, is_training: True,
                                                                                  anneal_val: anneal,
                                                                                  anneal_val_vae: anneal_vae,
                                                                                  batch_placeholder: args["batchsize"]})

                    to_save = [losses_val, save_val_ndcg, test_ndcgs, args, all_val_ndcg, user_matrix_vals, item_matrix_vals, imp_emb_values]

                if patience_counter >= patience:
                    running = False

                pickle.dump(to_save, open(savename, "wb"))

                print("patience", patience_counter, "/", patience, (datetime.datetime.now()))
                fprint(args["ofile"], " ".join([str(v) for v in ["patience", patience_counter, "/", patience, (datetime.datetime.now())]]))


if __name__ == "__main__":
    pool = mp.Pool(4)
    main()
    pool.close()
    pool.join()