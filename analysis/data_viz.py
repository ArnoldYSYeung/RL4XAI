"""
Class for data visualization, analysis, etc.

Date :                  February 29th, 2019

"""

import data_utils
import wallet
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats

from private_variables import *  # import variables "url" and "key" 

def calculate_cohens_d(count=[], mean=[], std=[]):
    """
    Calculate the Cohen's d effect size value for 2 distributions.
    Arguments:
        - count (list[int, int]):           number of samples for each distribution
        - mean (list[float, float]):        mean value for each distribution
        - std (list[float, float]):         standard devs for each distribution
    Returns:
        - Cohen's d value (float)
    """    
    pooled_std = np.sqrt(((count[0]-1.0)*std[0]**2 + (count[1]-1.0)*std[1]**2) / (count[0]+count[1]-2))
    
    d = abs(mean[0] - mean[1]) / pooled_std
    
    return d

class Batch():
    """
    Contains the data of one or more MTurk batches.
    """
    
    def __init__(self, url, auth_key, num_iter, mooclet, mturk_batch=[], include_testerbot=None):
        self._url = url
        self._auth_key = auth_key
        self.mooclet = mooclet
        self.num_iter = num_iter
        self.include_testerbot = include_testerbot
        
        if mturk_batch == []:
            raise ValueError("No batch selected.")
        self.mturk_batch = mturk_batch

        # Other attributes
        self.learner_ids = None
        self.values = None
        self.actions = None
        
    
    def __len__(self):
        raise NotImplementedError() 
    
    def get_data(self):
        """
        Retrieves the data for all learners within this batch.
        Arguments:
        """
        requirements = {'mooclet': self.mooclet}
        
        values = data_utils.get_values_meet_requirements(self._url+'value', requirements, 
                                                    variables=[], auth_key=self._auth_key)
    
        # filter out learners with batch not in self.mturk_batch
        if self.learner_ids is None:
            self.learner_ids = self.get_learner_ids()      
        
        for learner_id in list(values):
            if learner_id is None or learner_id[:8] not in self.learner_ids:
                del values[learner_id]
        
        self.values = values
        
        return values
        
    def save_data(self, filename):
        
        if self.values is None:
            raise ValueError("Can't find any data.")
        
        if filename:
            with open(filename, 'w') as file:
                json.dump(self.values, file)
        
        return True
    
    def load_data(self, filename):
        
        if filename:
            with open(filename, 'r') as file:
                self.values = json.load(file)
        
        return True
    
    def save_learner_ids(self, filename):
        
        if self.learner_ids is None:
            raise ValueError("Can't find learner ids.")
            
        if filename:
            with open(filename, 'w') as file:
                json.dump(self.learner_ids, file)
        
        return True
    
    def load_learner_ids(self, filename):
        
        if filename:
            with open(filename, 'r') as file:
                self.learner_ids = json.load(file)
        
        return True
    
    def save_actions(self, filename):
        
        if self.actions is None:
            raise ValueError("Can't find actions.")
            
        if filename:
            with open(filename, 'w') as file:
                json.dump(self.actions, file)
    
    def load_actions(self, filename):
        
        self.actions = {}
        if filename:
            with open(filename, 'r') as file:
                actions = json.load(file)
                
        for action in actions:
            self.actions[int(action)] = actions[action]
        
        return True
    
    def get_wallet(self):
        """
        Get the max wallet values for all learners in mturk_batch.
        Returns:
            - dictionary {learner_id: amount, }
        """
        data = {}
        for batch in self.mturk_batch:
            print("Getting wallet for batch", str(batch), "...")
            new_data = wallet.get_wallet_for_workerid(self._url+"value", batch, 
                                                  self.mooclet, self._auth_key)
            data = {**data, **new_data}
            
        return data
    
    
    def get_learner_ids(self):
        """
        Gets a dictionary of all learner ids which matches mturk_batch and mooclet.
        Arguments:
        Returns:
            - dictionary {learner_id: mturk_batch value, }

        """        
        print("Getting learners...")
        requirements = {'mooclet': self.mooclet}
        learners = data_utils.get_values_meet_requirements(self._url+'value', requirements, 
                                                     variables=['mturk_batch'], auth_key=self._auth_key)

        eligible_learners = {}
        # remove entries where mturk_batch not in self.mturk_batch
        print("Finding matching mturk_batch...")
        for learner_id in learners:
            if learners[learner_id]['mturk_batch']['value'] in self.mturk_batch:
                eligible_learners[learner_id] = learners[learner_id]['mturk_batch']['value']

        # get testerbot_ids
        if self.include_testerbot is False or self.include_testerbot is True:
            print("Searching for testerbots...")
            testerbot_ids = {}
            learner_data, _ = data_utils.get_json_response(self._url+"learner", 
                                                        auth_key = self._auth_key)
            for learner_entry in learner_data:
                
                if str(learner_entry['learner_id']) in eligible_learners: 
                    if len(learner_entry['name']) >= 4 \
                    and learner_entry['name'][:4] == "test":
                        testerbot_ids[learner_entry['learner_id']] = learner_entry['name']
                        
        # remove testerbots
        if self.include_testerbot is False:
            for testerbot_id in testerbot_ids:
                if str(testerbot_id) in eligible_learners:
                    del eligible_learners[str(testerbot_id)]
                else:
                    print("Cannot find testerbot_id: ", testerbot_id)
        
        # extract out testerbots
        elif self.include_testerbot is True:
            testerbot_learners = {}
            for testerbot_id in testerbot_ids:
                if str(testerbot_id) in eligible_learners:
                    testerbot_learners[str(testerbot_id)] = eligible_learners[str(testerbot_id)]
                else:
                    print("Cannot find testerbot_id: ", testerbot_id)
            eligible_learners = testerbot_learners
        
        self.learner_ids = eligible_learners
        
        return eligible_learners
    
    def filter_for_learners(self, values, learners):
        """
        Filters out values which have learners that are not listed.
        """
        raise NotImplementedError()
    
    def normalize_values(self, variables=[], baseline_iter_num = 1):
        """
        Calculates the normalized values of variables for each learner and adds
        to self.values.
        Adds the entry variable+"_norm": {'value': norm_value} into each learner
        Arguments:
            - variables (list[str, ]):          variables to normalize
            - baseline_iter_num (int):          iteration number for baseline
        """
        
        if self.values is None:
            self.values = self.get_data()
        
        if not isinstance(variables, list):
            raise ValueError("Input variable not a list.")
        
        base_values = {}
        for id in self.learner_ids:
            base_id = id+"_"+str(baseline_iter_num)
            
            if base_id in self.values:
                base_values[base_id] = self.values[base_id]
            
                for i in range(baseline_iter_num, self.num_iter+1):
                    learner_id = id+"_"+str(i)
                    if learner_id in self.values:
                        
                        if variables == []:
                            keys = self.values[learner_id].keys()
                            variables = list(keys)
                        
                        for variable in variables:
                            if variable in self.values[learner_id] and variable in base_values[base_id]:
                                # only if int or float
                                value = self.values[learner_id][variable]['value']
                                if isinstance(value, int) or isinstance(value, float):
                                    norm_value = value - base_values[base_id][variable]['value']
                                    # add normalized value to self.values
                                    self.values[learner_id][variable+str('_norm')] = {'value': norm_value}
                

    def get_progress_score(self, variable):
        """
        Creates a DataFrame (learner_ids, number of iterations) where each row is the value
        of variable for a learner each iteration.
        Arguments:
            - variable (str) :                     name of variable to get progress of
        Returns:
            DataFrame where each row is the progress score of a learner for a variable
        """
        
        if self.values is None:
            self.values = self.get_data()
        
        # get values
        progress = {}
        for id in self.learner_ids:
            progress[id] = []
            for i in range(1, self.num_iter+1):
                learner_id = id + "_" + str(i)
                if learner_id not in self.values:
                    progress[id].append(np.NaN)
                elif variable not in self.values[learner_id]:
                    progress[id].append(np.NaN)
                else:
                    progress[id].append(self.values[learner_id][variable]['value'])
                    
        # convert progress to DataFrame
        df = pd.DataFrame.from_dict(progress).transpose()
        df.columns=list(range(1, self.num_iter+1))
        return df
    
    def plot_progress_scores(self, score_names=[], mean=False, err='std', plot=True, normalized=False):
        """
        Plot multiple scores over time for learner_ids.
        Arguments:
            - score_names (lst[str, ]):      list of variable names to get scores of
            - mean (boolean):                whether to plot the mean of all individuals or separately
            - err (str) :                    type of error bar to plot (None, 'std', 'se')
            - plot (boolean):                whether to plot graph
            - normalized (boolean):          whether to normalize scores or not
        Returns:
            if mean == true:
                dictionaries containing DataFrames for each score mean and err with score_name as key
            if mean == false:
                dictionary containing DataFrames for each score with score_name as key
        """
        
        if self.values is None:
            self.values = self.get_data()
        
        scores = {}
        if mean:
            scores_err = {}
            
        for score_name in score_names:
            scores[score_name] = self.get_progress_score(score_name)

            title = str(score_name)
            
            if normalized:
                scores[score_name] = scores[score_name].sub(scores[score_name][1], axis=0)
                title = 'Normalized '+title

            if mean:
                if err == 'std':
                    scores_err[score_name] = scores[score_name].std()
                elif err == 'se':
                    scores_err[score_name] = scores[score_name].sem()
                scores[score_name] = scores[score_name].mean()

            if plot:
                if mean and err is not None:
                    ax = scores[score_name].transpose().plot.line(legend=False, 
                                                                  yerr=scores_err[score_name],
                                                                  capsize=5.0,
                                                                  markeredgewidth=2)
                else:
                    ax = scores[score_name].transpose().plot.line(legend=False)
                ax.set_title(title)
                ax.set_xlabel('Iteration #')
                ax.set_ylabel('Score')

        if mean and err is not None:
            return scores, scores_err
        else:
            return scores
    
    def get_score_df(self, variables=[], iter_nums=[]):
        """
        Converts values for the specified variables into a Pandas dataframe.
        Arguments:
            - variables (list[str, ]):          variables to include in dataframe
            - iter_nums (list[int, ]):          iterations to include (empty = all iterations)
        Returns:
            - Pandas dataframe (# samples x # variables)
        """
        
        if self.values is None:
            self.values = self.get_data()
        
        # create empty dataframe
        df = {'learner_id': []}
        for variable in variables:
            df[variable] = []
        
        for id in self.learner_ids:
            
            if iter_nums == []:
                iter_nums = list(range(1, self.num_iter+1))
                
            for i in iter_nums:
                learner_id = id + "_" + str(i)
                if learner_id in self.values:
                    missing_variable_flag = False            # flag to check whether a variable is missing
                    for variable in variables:
                        if variable not in self.values[learner_id]:
                            #print("Variable "+str(variable)+" not found for learner "+str(learner_id)+".")
                            missing_variable_flag = True
                            break
                    
                    if missing_variable_flag is False:
                        df['learner_id'].append(learner_id)
                        for variable in variables:
                            value = self.values[learner_id][variable]['value']
                            df[variable].append(value)
        
        df = pd.DataFrame.from_dict(df)
        
        return df                   
    
    def correlate_scores(self, variables=[], iter_nums=[], scatterplot=True, corr_method='pearson'):
        """
        Calculate the correlation between 2 variables and plot the scatterplot.
        Arguments:
            - variables (lst[str, str]):        variables to find correlation of
            - iter_nums (list[int, ]):          iterations to include (empty = all iterations)
            - scatterplot (boolean):            whether to plot a scatterplot
            - corr_method (str):                correlation method
        Returns:
            - Pandas DataFrame of correlation matrix
        """
        
        if len(variables) != 2:
            raise ValueError("'variables' does not have 2 values.")

        df = self.get_score_df(variables, iter_nums)

        print(df)
        
        if scatterplot:
            ax = df.plot.scatter(x=variables[0], y=variables[1], c='DarkBlue')
            
        # calculate the correlation coefficient    
        return df.corr(method=corr_method)

    def get_effect_size(self, variable, iter_nums=[]):
        """
        Calculate the effect size (Cohen's d) between 2 iterations for a variable.
        https://www.youtube.com/watch?v=tTgouKMz-eI
        https://www.youtube.com/watch?v=IetVSlrndpI
        Cohen's d is a measure of the distance between means. (d can be loosely interpreted
        as the number of stds the means are away from each other)
        Arguments:
            - variable (str):                   the variable to get effect size for
            - iter_nums (list[int, int]):       iterations to get effect size of
        Returns:
            - Cohen's d as a numerical value (float)
        """

        if len(iter_nums) != 2:
            raise ValueError("'iter_nums' is not length 2.")
            
        mean = []
        std = []
        count = []
        df = {}
        
        #   calculate mean, std, and count for each iteration
        idx = 0
        for i in iter_nums:
            df[i] = self.get_score_df(variables = [variable], iter_nums = [i])
            mean.append(df[i][variable].mean())
            std.append(df[i][variable].std())
            count.append(df[i][variable].count())
            print("Iteration: "+str(i)+\
                  "   Mean: "+str(mean[idx])+\
                  "   StD: "+str(std[idx])+\
                  "   Count: "+str(count[idx]))
            idx += 1
        
        # calculate Cohen's d
        pooled_std = np.sqrt(((count[0]-1.0)*std[0]**2 + (count[1]-1.0)*std[1]**2) / (count[0]+count[1]-2))
        d = abs(mean[0] - mean[1]) / pooled_std

        return d
    
    def get_effect_of_action(self, variable, action_ids=[], err="std"):
        """
        Measures the difference (in score and effect size) after the intervention of 1 or more actions.
        Arguments:
            - variable (str) :              the variable to get the change in score for after action
            - action_ids (list[int, ]):     ids of actions to observe 
            - err (string):                 type of error to return ("std", "se")
        Returns:
            - DataFrame containing previous scores, current scores (after providing action), 
                and differences in score for each sample
            - list containing mean difference, std of difference, and number of observations
        """
        
        if action_ids == []:
            raise ValueError("'action_ids' is empty.")
        
        if self.values is None:
            self.values = self.get_data()

        #   store the values of score before and when action is applied
        df = {}
        df['previous_id'] = []
        df['current_id'] = []
        df['previous_score'] = []
        df['current_score'] = []
        df['difference'] = []
        
        for id in self.learner_ids:
            
            for i in range(1, self.num_iter+1):
                prev_learner_id = id+"_"+str(i-1)
                learner_id = id+"_"+str(i)
                
                #   baseline has no 'action_id' variable
                if i != 1 and learner_id in self.values and \
                    variable in self.values[learner_id] and \
                    self.values[learner_id]['action_id']['value'] in action_ids:
                    #   matching action id
                    prev_score = self.values[prev_learner_id][variable]['value']
                    curr_score = self.values[learner_id][variable]['value']
                    df['previous_id'].append(prev_learner_id)
                    df['current_id'].append(learner_id)
                    df['previous_score'].append(prev_score)
                    df['current_score'].append(curr_score)
                    df['difference'].append(curr_score-prev_score)
        
        df = pd.DataFrame.from_dict(df)
        mean = df['difference'].mean()
        count = df['difference'].count()
        
        if err == "std":
            err = df['difference'].std()
        elif err == "se":
            err = df['difference'].sem()
        else:
            err = None
        
        return df, [mean, err, count]
    
    def count_bots(self):
        """
        Counts the number of learners who triggered at least 1 bot flag.

        Returns:
            - number of learners who triggered at least 1 bot flag (int)
            - dictionary for each bot type containing tuples of 
                (learner_id, iteration) where each bot type occurs (dict{tuple(),})
                    
        """
        
        bot_dict = {}
        
        if self.values is None:
            self.values = self.get_data()
        
        count = 0
        
        for learner_iter in self.values:
            if "bot" in self.values[learner_iter]:
                id = learner_iter[:8]
                iteration = learner_iter[9:]
                bot_type = self.values[learner_iter]["bot"]["value"]
                if bot_type not in bot_dict:
                    bot_dict[bot_type] = [(id, iteration)]
                else:
                    bot_dict[bot_type].append((id, iteration))
                count += 1
        
        return count, bot_dict
    
    def count_progress(self, iteration):
        """
        Counts the number of learners who have completed a specific iteration.
        Arguments:
            - iteration (str, int):             the iteration number (int) or "pa" (str)
        Returns:
            - number of learners (int)
            - list of learner ids (list[int, ])
            - dictionary of values (dict{learner_id:dict{}, })
        """
        
        if isinstance(iteration, str):
            if iteration != "pa":
                raise ValueError("String but not 'pa'.")
        elif isinstance(iteration, int):
            if iteration > self.num_iter:
                raise ValueError("Iteration out of bounds.")
        
        if self.values is None:
            self.values = self.get_data()
        
        count = 0
        success_count = 0
        
        learner_list = []
        learner_dict = {}
        
        for id in self.learner_ids:
            learner_id = id+"_"+str(iteration)
            if learner_id in self.values:
                learner_list.append(id)
                learner_dict[learner_id] = self.values[learner_id]
                count += 1
        
        return count, learner_list, learner_dict
    
    def get_explanation_effect_df(self, variable, err="std"):
        """
        Gets a dataframe containing the mean difference in simulatability scores
            for each classification category.  These are compared for corresponding
            explanations and non-corresponding explanations.
        Arguments:
            - variable (str):                   name of variable to get effect of
        Returns:
            - DataFrame containing mean, std, count for corresponding and non-corresponding
                actions for each variable score
        """
        
        if self.actions is None:
            self.get_actions()
        
        action_effects = {'score': [], 'corr_mean': [], 'corr_err': [], 'corr_count': [], 
                  'uncorr_mean': [], 'uncorr_err': [], 'uncorr_count': [], 'effect_size': [],
                  'p_value': []}
                
        for action_id in self.actions:
            
            if action_id == 13:
                #   special case baseline iteration
                score_name = variable
            else:
                action_name = self.actions[action_id]
                class_category = action_name[-2:]
                
                if variable[-4:] == "norm":
                    score_name = variable[:-4]+class_category+"_norm" 
                else:
                    score_name = variable+"_"+class_category
            
            # get effect on score and corresponding actions
            df, stats = self.get_effect_of_action(score_name, action_ids=[action_id],
                                                 err=err)
            corr_mean, corr_err, corr_count = stats

            # get effect on score for non-corresponding actions
            non_corr_action_ids = list(self.actions.keys())
            non_corr_action_ids.remove(action_id)
            _, stats = self.get_effect_of_action(score_name, 
                                                 action_ids=non_corr_action_ids,
                                                 err=err)
            uncorr_mean, uncorr_err, uncorr_count = stats
            
            if err == "std":
                corr_std = corr_err
                uncorr_std = uncorr_err
            else:
                _, stats = self.get_effect_of_action(score_name, action_ids=[action_id],
                                                 err="std")
                _, corr_std, _ = stats
                
                _, stats = self.get_effect_of_action(score_name, 
                                                 action_ids=non_corr_action_ids,
                                                 err="std")
                _, uncorr_std, _ = stats
                
           
            action_effects['score'].append(score_name)
            action_effects['corr_mean'].append(corr_mean)
            action_effects['corr_err'].append(corr_err)
            action_effects['corr_count'].append(corr_count)
            action_effects['uncorr_mean'].append(uncorr_mean)
            action_effects['uncorr_err'].append(uncorr_err)
            action_effects['uncorr_count'].append(uncorr_count)
            action_effects['effect_size'].append(calculate_cohens_d([corr_count, uncorr_count],
                                                                    [corr_mean, uncorr_mean],
                                                                    [corr_std, uncorr_std]))
            action_effects['p_value'].append(ttest_ind_from_stats(corr_mean, 
                                                                  corr_std,
                                                                  corr_count,
                                                                  uncorr_mean,
                                                                  uncorr_std,
                                                                  uncorr_count)[1])
            
        action_effects = pd.DataFrame.from_dict(action_effects)
            
        return action_effects
    
    def get_actions(self, include_baseline=True):
        """
        Get the actions corresponding to the mooclet and store as an attribute.
        Returns:
            - self.actions (dict{int: str, }) :             dictionary of action_id (key) and action name 
        """
        
        action_dict = data_utils.get_version(url=self._url, auth_key=self._auth_key)
        
        self.actions = {}
        
        for action in action_dict.json()['results']:
            if action['mooclet'] == self.mooclet:
                self.actions[action['id']] = action['name']
        
        #   special case where baseline iteration is denoted as action_id 13
        if include_baseline:
            self.actions[13] = 'baseline'

        return self.actions
        
class Experiment():
    """
    An experiment of 2 batches (baseline and experimental).
    """
    
    def __init__(self, baseline, experimental):
        """
        Arguments:
            - baseline (Batch):             batch containing baseline data
            - experimental (Batch):         batch containing experimental data
        """
        
        self.baseline = baseline
        self.experimental = experimental
        self.num_iter = int(min(self.baseline.num_iter, self.experimental.num_iter))
    
    def compare_scores_per_iter(self, variable, iter_nums=[], err='se', plot=True, title=None, normalized=False):
        """
        Compare the scores of the 2 batches for each iteration.
        Arguments:
            - variable (str) :                  name of score to compare
            - iter_nums (list[int,]):           list of iterations to compare (all if None)
            - err (str):                        type of error bar to return and plot ('std', 'se')
            - plot (bool):                      whether to plot graph
            - normalized (bool):                whether to normalize scores to first iteration or not
        Returns:
            - mean of each
            - error of each
            - mean difference
            - effect size between means
            - p-value between means
        """
        
        scores = {}
        
        #   get standard deviation (used for effect size)
        if err != "std":
            _, baseline_std = self.baseline.plot_progress_scores(score_names=[variable], 
                                                                         mean=True, err="std", plot=False,
                                                                         normalized=normalized)
            _, exp_std = self.experimental.plot_progress_scores(score_names=[variable],
                                                                   mean=True, err="std", plot=False,
                                                                   normalized=normalized)
        
        baseline_mean, baseline_err = self.baseline.plot_progress_scores(score_names=[variable], 
                                                                         mean=True, err=err, plot=False,
                                                                         normalized=normalized)
        
        exp_mean, exp_err = self.experimental.plot_progress_scores(score_names=[variable],
                                                                   mean=True, err=err, plot=False,
                                                                   normalized=normalized)
        if err == "std":
            baseline_std = baseline_err
            exp_std = exp_err
            
        baseline_std = baseline_std[variable].to_list()
        exp_std = exp_std[variable].to_list()
                    
        #   count the number of samples for each iteration
        scores['baseline_count'] = []
        scores['exp_count'] = []
        for i in range(1, self.num_iter+1):
            scores['baseline_count'].append(self.baseline.count_progress(i)[0])
            scores['exp_count'].append(self.experimental.count_progress(i)[0])

        scores['baseline_mean'] = baseline_mean[variable].to_list()
        scores['baseline_err'] = baseline_err[variable].to_list()
        scores['exp_mean'] = exp_mean[variable].to_list()
        scores['exp_err'] = exp_err[variable].to_list()
    

        scores['mean_diff'] = (exp_mean[variable] - baseline_mean[variable]).to_list()
        
        
        scores['effect_size'] = calculate_cohens_d(count=[np.asarray(scores['exp_count']), np.asarray(scores['baseline_count'])],
                                                   mean=[np.asarray(scores['exp_mean']), np.asarray(scores['baseline_mean'])],
                                                   std=[np.asarray(exp_std), np.asarray(baseline_std)]
                                                   )
        
        #   compute the p_value for each iteration
        scores['p_value'] = []
        for iteration in range(len(scores['exp_mean'])):
            scores['p_value'].append(ttest_ind_from_stats(scores['exp_mean'][iteration],
                                                          exp_std[iteration],
                                                          scores['exp_count'][iteration],
                                                          scores['baseline_mean'][iteration],
                                                          baseline_std[iteration],
                                                          scores['baseline_count'][iteration])[1])
        
        scores = pd.DataFrame.from_dict(scores)
        
        if plot:
            if err is not None:
                    ax = scores['baseline_mean'].transpose().plot.line(legend=True, 
                                                                  yerr=scores['baseline_err'],
                                                                  capsize=5.0,
                                                                  markeredgewidth=2)
                    ax = scores['exp_mean'].transpose().plot.line(legend=True, 
                                                                  yerr=scores['exp_err'],
                                                                  capsize=5.0,
                                                                  markeredgewidth=2)
            else:
                ax = scores['baseline_mean'].transpose().plot.line(legend=False)
                ax = scores['exp_mean'].transpose().plot.line(legend=False)
            
            if title is None:
                ax.set_title(variable)
            else:
                ax.set_title(title)
            ax.set_xlabel('Iteration #')
            ax.set_ylabel('Score')
            ax.legend(loc='lower right')
        
        return scores
    
    
    def compare_scores_per_action(self, variable, err="std"):
        """
        Returns a DataFrame comparing the change in variable score for each action 
        (explanation), for both baseline and experimental batches.
        Arguments:
            - variable (str) :                      variable to compare
            - err (str) :                           error ("std", "se")
        Returns:
            - DataFrame containing score mean, std, and count for all actions for
                both baseline and experimental batches.  Also includes effect size
                and p-value for comparison between the two batches.
        """
        if self.baseline.actions is None:
            bl_actions = self.baseline.get_actions()
        if self.experimental.actions is None:
            exp_actions = self.experimental.get_actions()
        
        bl_actions = self.baseline.actions.copy()
        bl_actions.pop(13, None)                 #   remove baseline iteration
        exp_actions = self.experimental.actions.copy()
        exp_actions.pop(13, None)
        
        if bl_actions != exp_actions:
            print("Baseline actions: ", bl_actions)
            print("Experimental actions: ", exp_actions)
            raise ValueError("Actions mismatched.")
        
        actions = bl_actions
        
        df = {'name': [], 'action': [],
              'bl_mean': [], 'bl_err': [], 'bl_count': [],
              'exp_mean': [], 'exp_err': [], 'exp_count': [],
              'effect_size': [], 'p_value': []}
        
        for action in actions:
            # use get effect_of_action
            action_name = actions[action]
            _, bl_stats = self.baseline.get_effect_of_action(variable, 
                                                             action_ids=[action],
                                                             err=err)
            _, exp_stats = self.experimental.get_effect_of_action(variable,
                                                                  action_ids=[action],
                                                                  err=err)
            
            bl_mean, bl_err, bl_count = bl_stats
            exp_mean, exp_err, exp_count = exp_stats
            
            if err == "std":
                bl_std = bl_err
                exp_std = exp_err
            else:
                _, bl_stats = self.baseline.get_effect_of_action(variable, 
                                                             action_ids=[action],
                                                             err="std")
                _, exp_stats = self.experimental.get_effect_of_action(variable,
                                                                      action_ids=[action],
                                                                      err="std")
                
                _, bl_std, _ = bl_stats
                _, exp_std, _ = exp_stats
            
            df['name'].append(action_name)
            df['action'].append(action)      
            df['bl_mean'].append(bl_mean)
            df['bl_err'].append(bl_err)
            df['bl_count'].append(bl_count)
            df['exp_mean'].append(exp_mean)
            df['exp_err'].append(exp_err)
            df['exp_count'].append(exp_count)
            df['effect_size'].append(calculate_cohens_d(count=[exp_count, bl_count],
                                                   mean=[exp_mean, bl_mean],
                                                   std=[exp_std, bl_std]))
            df['p_value'].append(ttest_ind_from_stats(exp_mean, 
                                                    exp_std,
                                                    exp_count,
                                                    bl_mean,
                                                    bl_std,
                                                    bl_count)[1])
        
        df = pd.DataFrame.from_dict(df)
        return df
            
    def compare_scores_all_actions(self, variable, err='std'):
        """
        Returns a DataFrame comparing the change in variable score for all actions
        (explanation) combined, for both baseline and experimental batches.
        Arguments:
            - variable (str) :                      variable to compare
            - err (string):                         type of error to return ("std", "se")
        Returns:
            - DataFrame containing score mean, std, and count for all actions for
                both baseline and experimental batches.  Also includes effect size
                and p-value for comparison between the two batches.
        """
        
        if self.baseline.actions is None:
            bl_actions = self.baseline.get_actions()
        if self.experimental.actions is None:
            exp_actions = self.experimental.get_actions()
        
        bl_actions = self.baseline.actions.copy()
        bl_actions.pop(13, None)                 #   remove baseline iteration
        exp_actions = self.experimental.actions.copy()
        exp_actions.pop(13, None)
        
        if bl_actions != exp_actions:
            print("Baseline actions: ", bl_actions)
            print("Experimental actions: ", exp_actions)
            raise ValueError("Actions mismatched.")
        
        actions = bl_actions
        
        df = {'bl_mean': [], 'bl_err': [], 'bl_count': [],
              'exp_mean': [], 'exp_err': [], 'exp_count': [],
              'effect_size': [], 'p_value': []}
        
        _, bl_stats = self.baseline.get_effect_of_action(variable, action_ids=actions, err=err)
        _, exp_stats = self.experimental.get_effect_of_action(variable, action_ids=actions, err=err)
        
        bl_mean, bl_err, bl_count = bl_stats
        exp_mean, exp_err, exp_count = exp_stats
        
        #   get standard deviation
        if err != 'std':
             _, bl_stats = self.baseline.get_effect_of_action(variable, action_ids=actions, err='std')
             _, exp_stats = self.experimental.get_effect_of_action(variable, action_ids=actions, err='std')
             _, bl_std, _ = bl_stats
             _, exp_std, _ = exp_stats
        else:
            exp_std = exp_err
            bl_std = bl_err
            
        df['bl_mean'] = bl_mean
        df['bl_err'] = bl_err
        df['bl_count'] = bl_count
        df['exp_mean'] = exp_mean
        df['exp_err'] = exp_err
        df['exp_count'] = exp_count
        df['effect_size'].append(calculate_cohens_d(count=[exp_count, bl_count],
                                                   mean=[exp_mean, bl_mean],
                                                   std=[exp_std, bl_std]))
        df['p_value'].append(ttest_ind_from_stats(exp_mean, 
                                                exp_std,
                                                exp_count,
                                                bl_mean,
                                                bl_std,
                                                bl_count)[1])
        
        df = pd.DataFrame.from_dict(df)
        return df
    
    
    def plot_effect_sizes_per_iter(self, variable, baseline_iter_num=None, title=None, plot=True):
        """
        Plot the effect sizes of each iteration relative to the baseline iteration
        for both the baseline batch and the experimental batch.
        Arguments:
            - variable (str) :                  name of variable to get effect size of
            - baseline_iter_num (int) :         iteration number to use as baseline 
                                                    (if None, compare to previous iteration)
            - title (str) :                     title of plot
        """
        
        baseline_effect_sizes = []
        exp_effect_sizes = []
        
        iterations = list(range(1, self.num_iter+1))
        if baseline_iter_num is not None:
            iterations.remove(baseline_iter_num)
        else:
            iterations.remove(1)        # first iteration does not have previous iteration
        
        for i in iterations:
            if baseline_iter_num:
                effect_size = self.baseline.get_effect_size(variable, 
                                                            iter_nums=[baseline_iter_num, i])
            else:
                effect_size = self.baseline.get_effect_size(variable,
                                                            iter_nums=[i-1, i])
            
            baseline_effect_sizes.append(effect_size)
            
            if baseline_iter_num:
                effect_size = self.experimental.get_effect_size(variable, 
                                                        iter_nums=[baseline_iter_num, i])
            else:
                effect_size = self.experimental.get_effect_size(variable,
                                                            iter_nums=[i-1, i])
            exp_effect_sizes.append(effect_size)
        
        if plot:
            # Set position of bar on X axis
            bar_width = 0.25
            r1 = iterations
            r2 = [x + bar_width for x in r1]
            
            # Make the plot
            plt.bar(r1, baseline_effect_sizes, color='#377eb8', width=bar_width, label='baseline')
            plt.bar(r2, exp_effect_sizes, color='#ff7f00', width=bar_width, label='exp')
            
            #plt.bar(r1, baseline_effect_sizes, color='#7f6d5f', width=bar_width, label='baseline')
            #plt.bar(r2, exp_effect_sizes, color='#557f2d', width=bar_width, label='exp')
            
            plt.xlabel("Iteration #")
            plt.ylabel("Simulatability Score Effect Size (Cohen's d)")
            plt.legend()
        
            if title is None:
                plt.title(variable+"_effect_size")
            else:
                plt.title(title)
        
            plt.show()
        
        return baseline_effect_sizes, exp_effect_sizes
    
    def plot_all_per_iter(self, variable, baseline_iter_num=1, title=None, 
                          ylim=None, ylim2=None, bar_labels=None, line_labels=None, 
                          save_file=None):
        """
        Plots combined bar (effect size) and line (variable mean) scores across all
        iterations for both batches.
        Arguments:
            - variable (str) :                  name of variable
            - baseline_iter_num (int) :         iteration # to use as baseline
            - title (str) :                     title of plot
            - ylim (list[float, float]):        limits of primary y-axis (line)
            - ylim2 (list[float, float]):       limits of secondary y-axis (bar)
            - bar_labels (list[str, str]):      labels for bars for each batch (baseline, experimental)
            - line_labels (list[str, str]):     labels for lines for each batch (baseline, experimental)
            - save_file (str):                  filename to save plot in (if None, does not save)

        """
        
        scores = self.compare_scores_per_iter(variable)
        bl_effect_sizes, exp_effect_sizes = self.plot_effect_sizes_per_iter(variable, 
                                                                            baseline_iter_num)
        
        iterations = list(range(0, self.num_iter))

        ax = plt.figure()
        
        # Set position of bar on X axis
        bar_width = 0.25
        r1 = iterations
        r2 = [x + bar_width for x in r1]
        
        if baseline_iter_num is not None:
            bl_effect_sizes.insert(baseline_iter_num-1, 0)
            exp_effect_sizes.insert(baseline_iter_num-1, 0)
        else:
            bl_effect_sizes.insert(0, 0)
            exp_effect_sizes.insert(0, 0)
        
        effect_df = {'iteration': r1, 'bl': bl_effect_sizes, 'exp': exp_effect_sizes}
        effect_df = pd.DataFrame.from_dict(effect_df)
        
        ax = effect_df.plot(x="iteration", y=["bl", "exp"], kind="bar", rot=0, 
                            alpha=0.5, label=bar_labels)
                
        plt.xlabel("Iteration #")
        plt.ylabel("Effect Size")
        
        ax2 = ax.twinx()
                
        ax2 = scores['baseline_mean'].transpose().plot.line(legend=False, 
                                                      yerr=scores['baseline_err'],
                                                      capsize=5.0,
                                                      markeredgewidth=2,
                                                      label=line_labels[0]
                                                      )

        ax2 = scores['exp_mean'].transpose().plot.line(legend=False, 
                                                      yerr=scores['exp_err'],
                                                      capsize=5.0,
                                                      markeredgewidth=2,
                                                      label=line_labels[1]
                                                      )
        
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()
        
        ax2.set_ylabel('Score')
        
        plt.xlim((min(iterations)-bar_width, self.num_iter-1+bar_width))
        
        if ylim is not None:
            ax2.set_ylim(ylim[0], ylim[1])
        if ylim2 is not None:
            ax.set_ylim(ylim2[0], ylim2[1])
            
        if title is not None:
            ax.set_title(title)
        
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h2+h1, l2+l1, loc='upper left', fontsize="small")
        
        if save_file is not None:
            plt.savefig(save_file)
        
        plt.show()
    
    def plot_all_per_iter2(self, variable, baseline_iter_num=1, title=None, 
                          ylim=None, ylim2=None, bar_labels=None, line_labels=None, 
                          legend=True, save_file=None):
        """
        Plots combined bar (variable mean) and line (effect size) scores across all
        iterations for both batches.
        Arguments:
            - variable (str) :                  name of variable
            - baseline_iter_num (int) :         iteration # to use as baseline
            - title (str) :                     title of plot
            - ylim (list[float, float]):        limits of primary y-axis (line)
            - ylim2 (list[float, float]):       limits of secondary y-axis (bar)
            - bar_labels (list[str, str]):      labels for bars for each batch (baseline, experimental)
            - line_labels (list[str, str]):     labels for lines for each batch (baseline, experimental)
            - legend (bool):                    whether to include legend
            - save_file (str):                  filename to save plot in (if None, does not save)

        """
        
        scores = self.compare_scores_per_iter(variable)
        bl_effect_sizes, exp_effect_sizes = self.plot_effect_sizes_per_iter(variable, 
                                                                            baseline_iter_num)
        
        iterations = list(range(0, self.num_iter))
        plt.close()
        ax = plt.figure()
        
        # Set position of bar on X axis
        bar_width = 0.25
        r1 = iterations
        r2 = [x + bar_width for x in r1]
        
        if baseline_iter_num is not None:
            bl_effect_sizes.insert(baseline_iter_num-1, 0)
            exp_effect_sizes.insert(baseline_iter_num-1, 0)
        else:
            bl_effect_sizes.insert(0, 0)
            exp_effect_sizes.insert(0, 0)
        
        effect_df = {'iteration': r1, 'bl': bl_effect_sizes, 'exp': exp_effect_sizes}
        effect_df = pd.DataFrame.from_dict(effect_df)
        
        print(scores[["baseline_err", "exp_err"]])
        
        ax = scores.plot(y=["baseline_mean", "exp_mean"], kind="bar", rot=0, 
                            alpha=0.5, label=bar_labels, 
                            yerr=scores[["baseline_err", "exp_err"]].values.T,
                            capsize=5.0, error_kw=dict(ecolor='gray'))
        
        plt.xlabel("Iteration #")
        plt.ylabel("Score")
        
        ax2 = ax.twinx()
                
        ax2 = effect_df['bl'].transpose().plot.line(legend=False, 
                                                      label=line_labels[0]
                                                      )

        ax2 = effect_df['exp'].transpose().plot.line(legend=False, 
                                                      label=line_labels[1]
                                                      )
        
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        
        ax2.set_ylabel('Effect Size')
        
        plt.xlim((min(iterations)-bar_width, self.num_iter-1+bar_width))
        
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if ylim2 is not None:
            ax2.set_ylim(ylim2[0], ylim2[1])
            
        if title is not None:
            ax.set_title(title)
        
        if legend:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='upper left', fontsize="small")
        else:
            ax.get_legend().remove()
        
        if save_file is not None:
            plt.savefig(save_file, format=save_file[-3:], dpi=1200, 
                        bbox_inches = "tight")
        
        plt.show()
            