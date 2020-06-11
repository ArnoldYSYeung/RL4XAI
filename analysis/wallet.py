"""
Worker ID and Payment Amount Matcher and Automated Payer

Date :              January 6th, 2020

"""
import boto3
import pandas as pd
import numpy as np
import re
from datetime import datetime

from data_utils import get_json_response
from private_variables import *

def get_wallet_for_workerid(url, mturk_batch, mooclet, auth_key=""):
    """
    Returns the wallet amount for each worker ID elligible from the
    inputted requirements.
    Arguments:
        url (str) :                       URL for MOOCLET database (end with "value")
        mturk_batch (int) :               code for mturk_batch
        mooclet (int) :                   mooclet id
    """
    length_of_id = 8
    
    #   get all values
    values, _ = get_json_response(url, auth_key)
    
    #   for each learner
    payment = {}                #   dictionary of dictionaries
    
    for value in values:
        
        if value['mooclet'] == mooclet:
        
            if 'learner' in value and value['learner'] != None \
                and len(value['learner']) >= length_of_id:
                    
                #   convert first 6 digits of workerid to string
                workerid = value['learner'][0:length_of_id]        #   get first 6 digits
                
                #   if not existing workerid
                if workerid not in payment:
                    #   create new worker
                    payment[workerid] = {'wallet': 0}
                    
                if value['variable'] == 'mturk_batch':
                    payment[workerid]['mturk_batch'] = value['value']
                if value['variable'] == 'wallet':
                    if value['value'] > payment[workerid]['wallet']:
                        payment[workerid]['wallet'] = value['value']

    mturk_batch_wallet = {}
    for workerid in payment:
        if 'mturk_batch' in payment[workerid] and \
            payment[workerid]['mturk_batch'] == mturk_batch:
            mturk_batch_wallet[workerid] = payment[workerid]['wallet']          
            
    return mturk_batch_wallet


def pay_workers(hit_id, df, test=True):
    """
    Connects to MTurk server and pay workers for a specific HIT
    Arguments:
        - hit_id (str) :                HIT ID found in MTurk batch interface
        - df (pd.DataFrame) :           dataframe containing the worker id, approval, payment amount, and feedback
        - test (bool) :                 whether to operate in test mode (will not actually pay workers)
    """
    
    approve_feedback = "Thank you for participating in this study."
    bonus_feedback = "Additional payment for study progress."
    
    client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
        )
        
    response = client.list_assignments_for_hit(
        HITId=hit_id,
        AssignmentStatuses=['Submitted', 'Approved'],
        MaxResults=10,
        )
     
    assignments = response['Assignments']
    
    if 'NextToken' in response:
        next_token = response['NextToken']
    
        while response['Assignments'] != []:        
            response = client.list_assignments_for_hit(
                HITId=hit_id,
                NextToken=next_token,
                AssignmentStatuses=['Submitted', 'Approved'],
                MaxResults=10,
                )
        
            assignments += response['Assignments']
        
            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                break
    
    print("There are "+str(len(assignments))+" assignments.")
    
    results = {'Date': [], 'AssignmentID': [], 'WorkerID': [], 'LearnerID': [],
               'Approve': [], 'Bonus': [], 'BonusToken': [],
               'Error': []}
    
    #   calculate if have enough money to pay all
    account_balance = float(client.get_account_balance()['AvailableBalance'])
    amount_sum = df['amount'].sum()
    bonus_sum = df['Bonus'].sum()

    print("Total payment amount is "+str(amount_sum)+".")
    print("You have "+str(account_balance)+" in your account to pay for "+
          str(bonus_sum)+" total bonus payment.")
    
    if bonus_sum > account_balance:
        print("Payment cannot be completed.  Bonus amount is less than account balance.")
        return
    else:
        print("You will have "+str(account_balance-bonus_sum)+" remaining.")
    
    if test:
        print("You are in test mode.")
    else:
        print("WARNING: YOU ARE NOT IN TEST MODE.")
    
    entry = input("Press 'y' to proceed: ")
    if entry != "y":
        print("Payment cancelled.")
        return
    else:
        print("Payment continuing...")
        
    for assignment in assignments:
        
        assignment_id = assignment['AssignmentId']
        worker_id = assignment['WorkerId']
        assignment_status = assignment['AssignmentStatus']
        
        results['Date'].append(datetime.now().strftime("%Y%m%d_%H:%M:%S"))
        results['AssignmentID'].append(assignment_id)
        results['WorkerID'].append(worker_id)
        encountered_error = False

        #   make sure worker_id is available
        if worker_id not in df['WorkerId'].values:
            print(worker_id+" not found.")
            encountered_error = True
            results['LearnerID'].append(None)
            results['Approve'].append(None)
            results['Bonus'].append(None)
            results['BonusToken'].append(None)
            
        else:
            # get learner_id
            learner_id = df.loc[df['WorkerId'] == worker_id, 'Answer.surveycode'].iloc[0]
            results['LearnerID'].append(learner_id)
        
            # get approve or not from inputted table (0 or 0.25)
            approve = df.loc[df['WorkerId'] == worker_id, 'Approve'].iloc[0]
            
            # get reject feedback and bonus feedback
            if df.loc[df['WorkerId'] == worker_id, 'Bonus'].iloc[0] > 0:
                has_bonus = True
                bonus = df.loc[df['WorkerId'] == worker_id, 'Bonus'].iloc[0]
            else:
                has_bonus = False
            
            # create bonus UniqueRequestToken
            bonus_token = worker_id+"_"+assignment_id
            
            if approve:
                try:
                    if test is False:
                        #   This doesn't work.  Need to manually approve.
                        """
                        if assignment_status == "Submitted":
                            out = client.approve_assignment(AssignmentId=assignment_id, 
                                                            RequesterFeedback=approve_feedback, 
                                                            OverrideRejection=False)
                        """
                        pass
                    # print("Worker "+str(worker_id)+" has been approved.")
                    results['Approve'].append(True)
                except Exception as e:
                    print(e)
                    print("Worker "+str(worker_id)+" has approval error.")
                    results['Approve'].append(None)
                    encountered_error = True
            else:
                try: 
                    print("Worker "+str(worker_id)+" has NOT been approved.")
                    results['Approve'].append(False)
                except:
                    print("Worker "+str(worker_id)+" has non-approval error.")
                    results['Approve'].append(None)
                    encountered_error = True
                
            if approve and has_bonus:
                try:
                    if test is False:
                        out = client.send_bonus(WorkerId=worker_id, 
                                                BonusAmount=str(bonus), 
                                                AssignmentId=assignment_id, 
                                                Reason=bonus_feedback, 
                                                UniqueRequestToken=bonus_token)
                        pass
                    # print("Worker "+str(worker_id)+" has been sent "+str(bonus)+" bonus.")
                    results['Bonus'].append(bonus)
                    results['BonusToken'].append(bonus_token)
                except Exception as e:
                    print(e)
                    print("Worker "+str(worker_id)+" has bonus error.")
                    results['Bonus'].append(None)
                    results['BonusToken'].append(None)
                    encountered_error = True
            else:
                results['Bonus'].append(None)
                results['BonusToken'].append(None)
        
        results['Error'].append(encountered_error)
    
    print("All payment completed.")
    
    return results

def pay_batch(url, key, hit_id, batch, mooclet, csv_file, test=True, approve_pay=0.25):
    """
    Pay bonuses to workers.
    Arguments:
        - url (str) :               URL to MOOClet server
        - key (str) :               authentication key
        - hit_id (str) :            MTurk HIT ID
        - mooclet (int) :           MOOClet number
        - csv_file (str) :          file containing MTurk batch results from MTurk website
        - test (bool) :             whether to use test mode (no payments)
        - approve_pay (float) :     base approval payment amount
    Returns:
        - 
        
    """
    
    print("Getting the wallet...")
    wallet = get_wallet_for_workerid(url+"value", batch, mooclet, key)
    
    #   convert wallet to pd.DataFrame
    wallet_df = {'learner_id': [], 'amount': []}
    for learner_id in wallet:
        wallet_df['learner_id'].append(learner_id)
        wallet_df['amount'].append(wallet[learner_id])
    wallet_df = pd.DataFrame.from_dict(wallet_df)
    
    print("Reading .csv file...")
    submissions_df = pd.read_csv(csv_file)[['HITId', 'WorkerId', 'AssignmentId', 'Answer.surveycode']]
    #submissions_df['Answer.surveycode'] = submissions_df['Answer.surveycode'].astype(int).astype(str)
    
    # remove all characters aside from numerals
    submissions_df['Answer.surveycode'] =  [re.sub(r'[^0-9]','', str(x)) 
                                            for x in submissions_df['Answer.surveycode']]
    
    #   check if Hit_id matches
    if submissions_df['HITId'].nunique() != 1:
        raise ValueError("More than 1 HITId detected.")
    if submissions_df['HITId'].iloc[0] != hit_id:
        print("From file: ", submissions_df['HITId'].iloc[0])
        print("Input: ", hit_id)
        raise ValueError('Mismatched hit_id.')
    
    #   check if there are duplicate workers
    if submissions_df['WorkerId'].count() != submissions_df['WorkerId'].nunique():
        raise ValueError('There are duplicate workers.')
    
    #return submissions_df
    
    #   link learner id to worker id
    #   removes if can't find a match
    submissions_df = pd.merge(submissions_df, 
                              wallet_df.set_index('learner_id'), 
                              left_on='Answer.surveycode', 
                              right_index=True)
    
    #   we only want workers who match assignments and will be approved
    submissions_df['Approve'] = submissions_df['amount'] > 0

    submissions_df['Bonus'] = submissions_df['amount'].apply(lambda x: x - approve_pay if x > approve_pay else None)   
    
    reject_df = submissions_df[submissions_df['amount'] <= 0]
    accept_df = submissions_df[submissions_df['amount'] > 0]
    
    
    out = pay_workers(hit_id, submissions_df, test=test)

    now = datetime.now().strftime("%Y%m%d_%H%M")
    filename = now+"_test_batch_"+str(batch)+"_payment.csv"
    if test:
        #   save output pd.DataFrame as file
        filename = now+"_test_batch_"+str(batch)+"_payment.csv"
    else:
        filename = now+"_pay_batch_"+str(batch)+"_payment.csv"
        
    pd.DataFrame.from_dict(out).to_csv(filename)

    reject_df.to_csv(now+"_reject_batch_"+str(batch)+"_payment.csv")
    accept_df.to_csv(now+"_accept_batch_"+str(batch)+"_payment.csv")

    return out, accept_df, reject_df, wallet

def check_payment(hit_id, payment="bonus"):
    
    client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
        )
     
    if payment == "bonus":
        response = client.list_bonus_payments(
            HITId=hit_id,
            )
    elif payment == "assignment":
        pass
    
    assignments = response['BonusPayments']
    
    if 'NextToken' in response and payment == "bonus":
        next_token = response['NextToken']
        print(next_token)
    
        while response['BonusPayments'] != []:        
            response = client.list_bonus_payments(
                HITId=hit_id,
                NextToken=next_token,
                #MaxResults=10,
                )
        
            assignments += response['BonusPayments']
        
            if 'NextToken' in response and response['BonusPayments'] != []:
                next_token = response['NextToken']
            else:
                break
       
    return assignments

if __name__ == "__main__":
     pass

