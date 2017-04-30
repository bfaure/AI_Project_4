import sys,os,time
from copy import copy,deepcopy

discount=0.9

reward_dict={"s1":0.0,"s2":0.0,"s3":1.0,"s4":0.0}

trans_dict= {"s1":{ "a1":[["s1",0.2],["s2",0.8]],"a2":[["s1",0.2],["s4",0.8]]},
			 "s2":{ "a2":[["s2",0.2],["s3",0.8]],"a3":[["s2",0.2],["s1",0.8]]},
			 "s3":{ "a4":[["s2",1.0]],           "a3":[["s4",1.0]]},
			 "s4":{ "a1":[["s4",0.1],["s3",0.9]],"a4":[["s4",0.2],["s1",0.8]]}}

def write_utilities(util_dict,device=sys.stdout):
	device.write("Utilties:\n")
	for key,value in util_dict.items():
		device.write("%s:\t%0.5f\n"%(key,value))

def R(s):
	try:    return reward_dict[s]
	except: return -1

def T(si,a,sj):
	try:
		possible_actions=trans_dict[si][a]
		for s,p in possible_actions:
			if s==sj: return p 
	except: return -1

def U(s,prior_utils):
	best_action_util=None 
	for action,outcomes in trans_dict[s].items():
		action_util=0.0
		for s1,s1_prob in outcomes:
			action_util+=(s1_prob*prior_utils[s1])
		if best_action_util==None or action_util>best_action_util:
			best_action_util=action_util
			best_action=action
	return reward_dict[s]+(discount*best_action_util),best_action

# err = maximum error allowed in the utility of any state
def value_iteration(err=0.0001):
	util_log=open("value_iteration-utils.tsv","w")
	action_log=open("value_iteration-actions.tsv","w")
	meta_log=open("value_iteration-meta.txt","w")
	meta_log.write("Discount: %0.10f\n"%discount)
	meta_log.write("Error: %0.10f\n"%err)

	print("Discount: %0.5f | Error: %0.10f"%(discount,err))

	start_time=time.time()

	i=0
	u0={}
	u1={}
	for s,u in reward_dict.items(): # create initial utilies distribution
		util_log.write("%s%s"%(s,"\t" if i!=len(reward_dict)-1 else "\n"))
		action_log.write("%s%s"%(s,"\t" if i!=len(reward_dict)-1 else "\n"))
		u0[s]=0.0
		u1[s]=0.0
		i+=1

	d=0.0
	i=0
	while True:
		i+=1
		sys.stdout.write("\rOptimizing states %d | d: %0.8f"%(i,d))
		sys.stdout.flush()

		d=0.0
		u0=copy(u1)

		j=0
		for s in reward_dict.keys(): # iterate over each state name
			u1[s],action = U(s,u0)
			util_log.write("%0.10f%s"%(u1[s],"\t" if j!=len(reward_dict)-1 else "\n"))
			action_log.write("%s%s"%(action,"\t" if j!=len(reward_dict)-1 else "\n"))
			if abs(u1[s]-u0[s])>d: d=abs(u1[s]-u0[s])
			j+=1

		if d<(err*((1-discount)/discount)): break
		
	sys.stdout.write("\nTotal time: %0.10f\n"%(time.time()-start_time))
	meta_log.write("Total Time: %0.10f\n"%(time.time()-start_time))
	return u0

def main():
	value_iteration()


if __name__ == '__main__':
	main()