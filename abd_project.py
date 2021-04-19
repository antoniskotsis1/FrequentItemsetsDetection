from itertools import combinations
import pandas as pd
import time
from math import ceil
from matplotlib import pyplot as plt


def present_results():
    return '''===========================================================================
(C) Create baskets and dataframes from csv file [format: c , <file>]
(S) Save baskets to csv files [format: s ]
(L) Load baskets from csv files [format: l ]
---------------------------------------------------------------------------------------------------------------------------
(SON) Run son algorithm [format: son, <chunk_size>, <min_support>, <max_length>, <u>(sers) or <m>(ovies)]
(APR) Run apriory algorithm [format: apr, <min_support>, <max_length>, <u>(sers) or <m>(ovies)]
(X) Run exact counting algorithm and present results [format: x, <K_min>, <K_max>, <u> or <m>]
(A) List ALL frequent itemsets [format: a ]
(B) List BEST (most frequent) itemset(s) [format: b, <itemset size> or <all>]
(M) Show details of a particular MOVIE [format: m , < comma-sep. movie_ids >]
(U) Show details of particular USERS [format: u , <comma-sep. user_ids>, -d(optional; for details)]
(H) Print the HISTOGRAM frequent itemsets [format: h ,<itemset size>]
(O) ORDER frequent itemsets by increasing support [format: o, <size> or <all> ]
---------------------------------------------------------------------------------------------------------------------------
(E) EXIT [format: e]
!Most of the results will be exported as csv!
===========================================================================\n'''


def read_ratings(file_name):
    try:
        return pd.read_csv(file_name.strip())
    except FileNotFoundError:
        print('No such file' + file_name)


def read_movies(file_name):
    try:
        return pd.read_csv(file_name.strip())
    except FileNotFoundError:
        print('No such file' + file_name)


def create_user_basket(dataframe, threshold):
    """Creates the baskets for each user by iterating the dataframe as a vector.
    Only movies rated with rank >= threshold will be returned.
    """
    result = {}
    for i, j, r in zip(dataframe['userId'], dataframe['movieId'], dataframe['rating']):
        if float(r) >= threshold:
            if i not in result:
                result[int(i)] = []
            result[int(i)].append(int(j))
    return result


def create_movie_baskets(dataframe, threshold):
    """Creates the baskets for each movie by iterating the dataframe as a vector
    Only movies rated with rank >= threshold will be returned.
    """

    result = {}

    for i, j, r in zip(dataframe['movieId'], dataframe['userId'], dataframe['rating']):

        if float(r) >= threshold:
            if i not in result:
                result[int(i)] = []
            result[int(i)].append(int(j))

    return result


def exact_counting(K_min, K_max, baskets):
    result = {}
    while K_min <= K_max:
        for basket in baskets.values():
            args = [i for i in range(len(basket))]
            temp = []
            for i in combinations(args, r=K_min):
                for j in i:
                    temp.append(basket[j])
                if result.get(temp[0] if K_min == 1 else tuple(temp)):
                    result[temp[0] if K_min == 1 else tuple(temp)] += 1
                else:
                    result[temp[0] if K_min == 1 else tuple(temp)] = 1
                temp.clear()
        K_min += 1
    return result


def count_singletons(baskets):
    """Creates a dictionary with each movie as a key and a list of all baskets
    containing that key as value. The length of that list is the support for each key
    """
    counter_dict = dict()
    for basket_id, basket_items in baskets.items():
        for item in basket_items:
            if counter_dict.get(item):
                counter_dict[item].add(basket_id)
            else:
                counter_dict[item] = {basket_id}
    return counter_dict


def save_df_to_csv(df, file_name, caller):
    if caller == 'apr':
        support_dataframes_apr.append(df)
    elif caller == 'son':
        support_dataframes_son.append(df)

    df.to_csv(file_name)


def filter_singleton_candidates(c1, baskets, min_support, son):
    """Returns a list containing only the frequent items and also deletes
    those non frequent ones from the dictionary created at count_singletons
    """
    res = {}
    to_be_deleted = set()
    for candidate, support in c1.items():
        if min_support <= len(support):
            res[candidate] = len(support)
        else:
            # eliminate non frequent items from the original basket
            for i in support:
                if baskets.get(i):
                    to_be_deleted.add(candidate)
    if res and not son:
        support_df = pd.DataFrame(list(zip(res.keys(), res.values())), columns=['itemset', 'support'])
        save_df_to_csv(support_df, f'APR_1s_support_{min_support}min_sup.csv', 'apr')
    for i in to_be_deleted:
        del c1[i]
    return sorted(res)


def create_next_candidate(k, prev):
    """Implements self join of Lk-1 itemsets to generate next candidates"""
    if k == 2:
        li = [frozenset([i]) for i in prev]
    else:
        li = [frozenset(i) for i in prev]

    res = set([i.union(j) for i in li for j in li if len(i.union(j)) == k])

    return list([tuple(i) for i in res])


def count_and_filter_support(candidates, frequent_singletons_dict, k, min_support, son):
    """Counts the support for each itemset by computing the length of the intersection
    between every singleton in the set. Returns only the frequent itemsets as a list.

    """
    supports = []
    res = {}
    for pair in candidates:
        if k > 2:
            temp = frequent_singletons_dict[pair[0]].intersection(frequent_singletons_dict[pair[1]])
            for index in range(2, k):
                temp = temp.intersection(frequent_singletons_dict[pair[index]])
            if len(temp) >= min_support:
                supports.append(len(temp))
                res[tuple(sorted(pair))] = len(temp)
        else:
            support = len(frequent_singletons_dict[pair[0]].intersection(frequent_singletons_dict[pair[1]]))
            if support >= min_support:
                supports.append(support)
                res[tuple(sorted(pair))] = support
    if res and not son:
        support_df = pd.DataFrame(list(zip(res.keys(), res.values())), columns=['itemset', 'support'])
        save_df_to_csv(support_df, f"APR_{str(k)}s_support_{min_support}min_supp.csv", 'apr')
    return sorted((res.keys()))


def my_apriory(min_support, max_length, baskets, son=False):
    """Implementation of the arpiory algorithm. The algorithm stops if the num of
    passes reaches max_length or when no other candidates can be generated"""
    frequent_itemsets = list()
    passes = 1
    candidate_singles = count_singletons(baskets)
    l1 = filter_singleton_candidates(candidate_singles, baskets, min_support, son)
    frequent_itemsets.append(list(l1))
    passes += 1
    next_l = l1
    while passes <= max_length and len(next_l) >= passes:
        candidates = create_next_candidate(passes, next_l)
        next_l = count_and_filter_support(candidates, candidate_singles, passes, min_support, son)
        passes += 1
        frequent_itemsets.append(next_l)
    return frequent_itemsets


def get_chunk(baskets, start_index, current_index):
    basket_chunk = {}
    for i in list(baskets.keys())[start_index:current_index]:
        basket_chunk[i] = baskets.get(i)
    return basket_chunk


def get_reduced_support(original_sup, original_basket, chunk):
    return ceil(original_sup * (1 - ((len(original_basket) - len(chunk)) / len(original_basket))))


def fill_dictionary(original_dict, freq_itemsets):
    count = 0
    for freq_items in freq_itemsets:
        for element in freq_items:
            if count != 0:
                element = tuple(sorted(element))

            if original_dict.get(element):
                original_dict[element] += 1
            else:
                original_dict[element] = 1
        count += 1


def count_son_itemsets(son_dict, singletons, min_support):
    """Second pass of son algorithm. Counts the candidates to erase false positives"""
    res = {}
    for items in son_dict.keys():
        try:
            if len(items) > 2:
                temp = singletons[items[0]].intersection(singletons[items[1]])
                for index in range(2, len(items)):
                    temp = temp.intersection(singletons[items[index]])
                if len(temp) >= min_support:
                    res[items] = len(temp)
            else:
                support = len(singletons[items[0]].intersection(singletons[items[1]]))
                if support >= min_support:
                    res[items] = support
        except TypeError:
            if singletons.get(items) and len(singletons.get(items)) >= min_support:
                res[items] = len(singletons.get(items))
    return res


def SON(baskets, chunks_size, min_support, max_length):
    """Implementation of SON algorithm. First pass generates candidates """
    first_pass_dictionary = dict()
    for i in range(0, len(baskets), chunks_size):
        chunk = get_chunk(baskets, i, i + chunks_size)
        chunks_support = get_reduced_support(min_support, baskets, chunk)
        frequent_candidates = my_apriory(chunks_support, max_length, chunk, son=True)
        fill_dictionary(first_pass_dictionary, frequent_candidates)
    candidate_singles_second_pass = count_singletons(baskets)  # all singleton supports
    final = count_son_itemsets(first_pass_dictionary, candidate_singles_second_pass, min_support)
    return final


def clear_zeros(user, movie):
    for u_key in user.keys():
        user[u_key] = list(filter(lambda i: i != 0, user[u_key]))
    for m_key in movie.keys():
        movie[m_key] = list(filter(lambda i: i != 0, movie[m_key]))


def load_baskets_from_csv():
    try:
        user = pd.read_csv('user_baskets.csv').fillna(0)  # needed to load the baskets
        user = user.astype(int).to_dict(orient='list')
        movie = pd.read_csv('movie_baskets.csv').fillna(0)
        movie = movie.astype(int).to_dict(orient='list')
        clear_zeros(user, movie)
        print("Successfully loaded baskets from csv files")
        return user, movie

    except FileNotFoundError:
        print("No file(s) found to load baskets from!")


def list_all_frequent_itemsets():
    if apriory_result:
        print("-----APRIORI'S RESULT----")
        count = 1
        for i in apriory_result:
            print(f"{count}s itemsets found: {len(i)}")
            print(i)
            count += 1
    if sons_dict:
        count = 1
        print("-----SON'S RESULT----")
        for i in sons_result:
            print(f"{count}s itemsets found: {len(i)}")
            print(i)
            count += 1


def create_list_like_output_for_son(item_dict):
    """Creates a list of list for the output of son algorithm.
        First element of that list contains frequent singletons second frequent doubles etc.
    """
    result = []
    supports = []
    for i in range(len(item_dict)):
        result.append([])
        supports.append([])

    for size_, values in item_dict.items():
        for y in values:
            if not isinstance(y[0], int):
                result[size_ - 1].append(y[0])
                supports[size_ - 1].append(y[1])
            else:
                result[0].append(y[0])
                supports[0].append(y[1])
    count = 1
    for item, supp in zip(result, supports):
        df = pd.DataFrame(list(zip(item, supp)), columns=['itemset', 'support'])
        save_df_to_csv(df, 'SON' + str(count) + 's_support.csv', 'son')
        count += 1
    return result


def get_sons_most_freq_items(all_=False):
    best_items = {}
    all_items = {}
    for itemset, support in sons_dict.items():
        if not isinstance(itemset, int):
            if best_items.get(len(itemset)):
                all_items[len(itemset)].append((itemset, support))
                if support >= best_items.get(len(itemset))[-1]:
                    best_items[len(itemset)] = (itemset, support)
            else:
                best_items[len(itemset)] = (itemset, support)
                all_items[len(itemset)] = [(itemset, support)]
        else:
            if best_items.get(1):
                all_items[1].append((itemset, support))
                if support >= best_items.get(1)[-1]:
                    best_items[1] = (itemset, support)
            else:
                best_items[1] = (itemset, support)
                all_items[1] = [(itemset, support)]
    if all_:
        freq_items = create_list_like_output_for_son(all_items)
        return freq_items
    else:
        return best_items


def list_best_frequent_itemsets():
    global support_dataframes_apr

    if apriory_result:
        print("-----APRIORI'S RESULT-----")
        if len(support_dataframes_apr) == 0:
            print("No frequent items found..")
            return
        df = support_dataframes_apr[size - 1]
        support = df[df['support'] == df['support'].max()]
        print(f"Most frequent {size}s-itemset(s):\n {support.to_string(index=False)} ")

    if sons_dict:
        print("-----SON'S RESULT-----")
        if len(support_dataframes_son) == 0:
            print("No frequent items found..")
            return
        df = support_dataframes_son[size - 1]
        support = df[df['support'] == df['support'].max()]
        print(f"Most frequent {size}s-itemset(s):\n {support.to_string(index=False)} ")


def list_all_best_itemsets():
    if len(support_dataframes_apr) == 0:
        print("No frequent items found..")
    else:
        print("-----APRIORI'S RESULT-----")
        for i in range(len(support_dataframes_apr)):
            itemset_index = support_dataframes_apr[i]['support'].idxmax()
            itemset = support_dataframes_apr[i].loc[itemset_index]['itemset']
            support = support_dataframes_apr[i]['support'].max()
            print(f"Most frequent {i + 1}s-itemset: {itemset}, with support: {support}")

    if len(support_dataframes_son) == 0:
        print("No frequent items found..")
    else:
        print("-----SON'S RESULT-----")
        for i in range(len(support_dataframes_son)):
            itemset_index = support_dataframes_son[i]['support'].idxmax()
            itemset = support_dataframes_son[i].loc[itemset_index]['itemset']
            support = support_dataframes_son[i]['support'].max()
            print(f"Most frequent {i + 1}s-itemset: {itemset}, with support: {support}")


def show_details_for_movie(movie_id):
    try:
        if len(movie_id) == 1:
            var = movies_df[movies_df['movieId'] == int(movie_id[0])].to_string(index=False, header=False)
            print(var)
        else:
            for i in range(len(movie_id)):
                var = movies_df[movies_df['movieId'] == int(movie_id[i])].to_string(index=False, header=False)
                print(var)
    except ValueError:
        print("Invalid Syntax!! Format: m, <comma sep. movieIds>")


def get_user_info(id_):
    try:
        id_ = int(id_)
        res = my_ratings_df.loc[my_ratings_df['userId'] == id_]
        movies_seen = res['movieId'].count()
        avg_rating = res['rating'].sum() / res['rating'].count()
        print(f"User:{id_} | Seen: {movies_seen} movies | Average Rating: {avg_rating}")
    except AttributeError:
        print("Create baskets first!!")


def get_detailed_user_info(id_):
    try:
        id_ = int(id_)
        res = my_ratings_df.loc[my_ratings_df['userId'] == id_]
        details = res[['movieId', 'rating']]
        print(f"UserId: {id_}\nMovie,rating:\n{details.to_string(index=False, header=False)}")
    except AttributeError:
        print("Create baskets first!!")


def show_details_for_users(user_id, detailed=False):
    if detailed:
        if len(users) == 1:
            get_detailed_user_info(int(user_id[0]))
        else:
            for i in range(len(user_id)):
                get_detailed_user_info(user_id[i])
    else:
        if len(user_id) == 1:
            get_user_info(user_id[0])
        else:
            for i in range(len(user_id)):
                get_user_info(user_id[i])


def order_all_frequent_itemsets():
    if support_dataframes_apr:
        all_in_one = pd.concat(support_dataframes_apr)
        all_in_one = all_in_one.sort_values(by=['support'], ascending=True, ignore_index=True)
        print(all_in_one)
        save_df_to_csv(all_in_one, 'sorted_itemsets_all.csv', 'other')

    if support_dataframes_son:
        all_in_one = pd.concat(support_dataframes_son)
        all_in_one = all_in_one.sort_values(by=['support'], ascending=True, ignore_index=True)
        print(all_in_one)
        save_df_to_csv(all_in_one, 'sorted_itemsets_all.csv', 'other')


def order_frequent_itemsets():
    if support_dataframes_apr:
        df = support_dataframes_apr[int(options[1]) - 1]
        df = df.sort_values(by=['support'], ascending=True, ignore_index=True)
        print(df)
        save_df_to_csv(df, f'APR_{int(options[1])}s-itemset_sorted.csv', 'other')

    if support_dataframes_son:
        df = support_dataframes_son[int(options[1]) - 1]
        df = df.sort_values(by=['support'], ascending=True, ignore_index=True)
        print(df)
        save_df_to_csv(df, f'SON_{int(options[1])}s-itemset_sorted.csv', 'other')


def save_baskets_to_csv():
    if user_basket:
        start = time.time()
        pd.DataFrame.from_dict(user_basket, orient='index').transpose().to_csv('user_baskets.csv', index=False)
        pd.DataFrame.from_dict(movie_basket, orient='index').transpose().to_csv('movie_baskets.csv', index=False)
        end = time.time()
        print("Successfully saved dataframes to user_baskets.csv and movie_baskets.csv")
        print(f"Time taken: {end - start}")

    else:
        print("Baskets not found. Create them first!")


def create_baskets_and_dfs(score_threshold=3.5):
    try:
        start = time.time()
        my_ratings_dfs = read_ratings(file)
        movie_baskets = create_movie_baskets(my_ratings_dfs, score_threshold)
        user_baskets = create_user_basket(my_ratings_dfs, score_threshold)

        end = time.time()
        print(f"Successfully created baskets and dataframes from {file}."
              f"Time taken: {end - start} secs")
        return my_ratings_dfs, movie_baskets, user_baskets
    except TypeError as e:
        print("File not found!Try again.." + str(e))


def create_histograms(df):
    df['support'].plot.hist(bins=12, alpha=1)
    plt.title(f'Histogram of supports for frequent {size}s-itemsets')
    plt.xlabel('Support')
    plt.ylabel('Number of Frequent Itemsets')
    plt.tight_layout()
    plt.show()


def run_exact_counting():
    st = time.time()
    res = exact_counting(k_min, k_max, ex_basket)
    end = time.time()
    print(res)
    print("Successfully executed my apriory algorithm!!")
    print(f"Time taken: {end - st} secs")
    # df = pd.DataFrame.from_dict(res, orient='index', columns=['support'])
    # save_df_to_csv(df, f'exact_counting_fr_{k_min}_to_{k_max}.csv','other')

    return res


def run_apriory(basket):
    try:
        support_dataframes_apr.clear()
        min_support = int(options[1])
        max_length = int(options[2])
        start = time.time()
        fr = my_apriory(min_support, max_length, basket)
        end = time.time()
        print("Successfully executed my apriory algorithm!!")
        print(f"Time taken: {end - start} secs")
        return fr
    except (ValueError, IndexError) as e:
        print(e)
        print('Illegal format!!Try again: a, <min_support>, <max_length>, basket')
    pass


def run_son(basket):
    try:
        chunk_size = int(options[1])
        min_support = int(options[2])
        max_length = int(options[3])
        start = time.time()
        son_fr = SON(basket, chunk_size, min_support, max_length)
        end = time.time()
        print("Successfully executed my SON algorithm!!")
        print(f"Time taken: {end - start} secs")
        return son_fr
    except (ValueError, IndexError) as e:
        print(e)
        print('Illegal format!!Try again: a, <min_support>, <max_length>, basket')


if __name__ == '__main__':
    movies_df = read_movies('movies.csv')
    support_dataframes_apr = []
    support_dataframes_son = []
    my_ratings_df, movie_basket, user_basket = [], [], []
    apriory_result = []
    sons_dict = []
    sons_result = {}

    print(present_results())
    option = input('>>')
    while option[0].lower() != 'e':

        options = option.split(',')

        if options[0].lower() == 'apr':

            if user_basket:
                if options[-1].lower() == 'u':
                    apriory_result = run_apriory(user_basket)
                else:
                    apriory_result = run_apriory(movie_basket)
            else:
                print("create baskets first")

        elif options[0].lower() == 'c':

            try:
                file = options[1]
                my_ratings_df, movie_basket, user_basket = create_baskets_and_dfs()
            except IndexError:
                print("Illegal format!! Format: c, <file_name>")
            except TypeError:
                pass

        elif options[0].lower() == 's':

            try:
                save_baskets_to_csv()
            except IndexError:
                print("Illegal format!! Format: s")

        elif options[0].lower() == 'l':

            user_basket, movie_basket = load_baskets_from_csv()

        elif options[0].lower() == 'son':

            if user_basket:
                if options[-1].lower() != 'u':
                    sons_dict = run_son(movie_basket)
                else:
                    sons_dict = run_son(user_basket)
                sons_result = get_sons_most_freq_items(all_=True)
            else:
                print("create baskets first")

        elif options[0].lower() == 'a':

            list_all_frequent_itemsets()

        elif options[0].lower() == 'b':

            try:
                size = int(options[1])
                list_best_frequent_itemsets()
            except ValueError:
                if options[1] == 'all':
                    list_all_best_itemsets()

        elif options[0].lower() == 'h':

            size = int(options[1])
            if support_dataframes_apr:
                print("-----APRIORI'S RESULT-----")
                create_histograms(support_dataframes_apr[size - 1])
            if support_dataframes_son:
                print("-----SON'S RESULT-----")
                create_histograms(support_dataframes_son[size - 1])

        elif options[0].lower() == 'm':

            movies = options[1:]
            show_details_for_movie(movies)

        elif options[0].lower() == 'u':

            if options[-1].startswith('-'):
                users = options[1:-1]
                show_details_for_users(users, detailed=True)
            else:
                users = options[1:]
                show_details_for_users(users)

        elif options[0].lower() == 'x':
            k_min, k_max = int(options[1]), int(options[2])
            if options[-1].lower() == 'u':
                ex_basket = user_basket
            else:
                ex_basket = movie_basket
            run_exact_counting()
        elif options[0].lower() == 'o':
            if options[1] == 'all':
                order_all_frequent_itemsets()
            else:
                order_frequent_itemsets()
        continue_ = input("Do you wish to continue? <y> / <n>: ")
        if continue_.lower() == 'y':
            option = input(present_results() + '>>')
        else:
            break
