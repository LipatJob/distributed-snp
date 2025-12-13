#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "snp_model.hpp"
#include "error_check.hpp"
// basic file operations
#include <iostream>
#include <fstream>

using namespace std;


/** Allocation */
SNP_model::SNP_model(uint n, uint m, bool using_lib)
{
    this->using_lib = using_lib;
    this->step = 0;
    this->done_rules = false;
    // allocation in CPU
    this->n = n;  // number of neurons
    this->m = m;  // number of rules
    
    this->rule_index      = (int*)   malloc(sizeof(int)*(n+1)); // indeces of rules inside neuron (start index per neuron)
    this->rules.Ei        = (uint*)  malloc(sizeof(uint)*m); // Regular expression Ei of a rule
    this->rules.En        = (uint*)  malloc(sizeof(uint)*m); // Regular expression En of a rule
    this->rules.c         = (uint*)  malloc(sizeof(uint)*m); // LHS of rule
    this->rules.p         = (uint*)  malloc(sizeof(uint)*m); // RHS of rule
    this->rules.d         = (uint*)  malloc(sizeof(uint)*m); // RHS of rule
    this->rules.nid       = (uint*)   malloc(sizeof(uint)*(m)); // Index of the neuron where the rule is
    this->calc_next_trans = (bool*) malloc(sizeof(bool));

    // allocation in GPU
    
    cuda_check(cudaMalloc(&this->d_rule_index,    sizeof(int)*(n+1)));
    cuda_check(cudaMalloc(&this->d_rules.Ei,      sizeof(uint)*m));
    cuda_check(cudaMalloc(&this->d_rules.En,      sizeof(uint)*m));
    cuda_check(cudaMalloc(&this->d_rules.c,       sizeof(uint)*m));
    cuda_check(cudaMalloc(&this->d_rules.p,       sizeof(uint)*m));
    cuda_check(cudaMalloc(&this->d_rules.d,       sizeof(uint)*m));
    cuda_check(cudaMalloc(&this->d_rules.nid,     sizeof(uint)*m));
    cuda_check(cudaMalloc(&this->d_calc_next_trans, sizeof(bool)));

    // initialization (only in CPU, having updated version)
    
    memset(this->rule_index,    -1,  sizeof(int)*(n+1));
    this->rule_index[0]=0;
    memset(this->rules.Ei,      0,  sizeof(uint)*m);
    memset(this->rules.En,      0,  sizeof(uint)*m);
    memset(this->rules.c,       0,  sizeof(uint)*m);
    memset(this->rules.p,       0,  sizeof(uint)*m);
    memset(this->rules.d,       0,  sizeof(uint)*n);
    memset(this->rules.nid,     0,  sizeof(uint)*(m));

    if(using_lib){
        this->f_conf_vector     = (float*) malloc(sizeof(float)*n); // configuration vector (only one, we simulate just a computation)
        cuda_check(cudaMalloc(&this->df_conf_vector,   sizeof(float)*n));
        memset(this->f_conf_vector,   0,  sizeof(float)*n);
    }else{
        this->conf_vector     = (uint*) malloc(sizeof(uint)*n); // configuration vector (only one, we simulate just a computation)
        this->delays_vector = (uint*) malloc(sizeof(uint)*n); 
        cuda_check(cudaMalloc(&this->d_conf_vector,   sizeof(uint)*n));
        cuda_check(cudaMalloc(&this->d_delays_vector,   sizeof(uint)*n));
        memset(this->conf_vector,   0,  sizeof(uint)*n);
        memset(this->delays_vector,   0,  sizeof(uint)*n);
    }
   
    // memory consistency, who has the updated copy?
    this->cpu_updated = true;
    this->gpu_updated = false;    
}

/** Free mem */
SNP_model::~SNP_model()
{
    free(this->conf_vector);
    free(this->spiking_vector);
    free(this->trans_matrix);
    free(this->rule_index);
    free(this->rules.Ei);
    free(this->rules.En);
    free(this->rules.c);
    free(this->rules.p);
    free(this->rules.d);
    free(this->rules.nid);
    free(this->calc_next_trans);
    free(this->delays_vector);
    free(this->outfile);

    cudaFree(this->d_conf_vector);
    cudaFree(this->d_spiking_vector);
    cudaFree(this->d_trans_matrix);
    cudaFree(this->d_rule_index);
    cudaFree(this->d_rules.Ei);
    cudaFree(this->d_rules.En);
    cudaFree(this->d_rules.c);
    cudaFree(this->d_rules.p);
    cudaFree(this->d_rules.d);
    cudaFree(this->d_rules.nid);
    cudaFree(this->d_calc_next_trans);
    cudaFree(this->d_delays_vector);
}

__global__ void k_print_conf_v(uint *conf_v, int n){
    printf("Configuration vector(gpu memory)\n");
    for(int i=0; i<n; i++){
        printf("%d ",conf_v[i]);
    }
    printf("\n");
}

void SNP_model::print_conf_vector (ofstream *fs){
    //////////////////////////////////////////////////////
    assert(gpu_updated || cpu_updated);
    if (!cpu_updated) load_to_cpu();
    //////////////////////////////////////////////////////
    
    if(fs != NULL){
        *fs << "Configuration vector\n";
    }else{
        printf("Configuration vector\n");
    }
    
    if(using_lib){
        for(int i=0; i<n; i++){
            if(fs != NULL){
                *fs << f_conf_vector[i] << " ";
            }else{
                printf("%.1f ",f_conf_vector[i]);
            }   
        }

    }else{
        for(int i=0; i<n; i++){
            if(fs != NULL){
                *fs << conf_vector[i] << " ";
            }else{
                printf("%d ",conf_vector[i]);
            }   
        }

    }
    
    printf("\n");
}

void SNP_model::set_snpconfig (int verbosity_lv, int repetitions, char *outfile, bool count_time, bool get_mem_info){
    this->verbosity_lv = verbosity_lv;
    this->repetitions = repetitions;
    this->outfile = outfile;
    this->count_time = count_time;
    this->get_mem_info = get_mem_info;
}

void SNP_model::write_to_file(){
    ofstream myfile;
    if(repetitions == -1){
        myfile.open (this->outfile, ios::out | ios::trunc);
    }else{
        myfile.open(this->outfile, std::ios_base::app);
    }
    myfile<<"\nComputation performed in " << this->step << " steps\n";
    print_conf_vector(&myfile);
    myfile.close();
}

void SNP_model::set_spikes (uint nid, uint s)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) load_to_cpu();
    gpu_updated = false;
    //////////////////////////////////////////////////////

    if(using_lib){
        f_conf_vector[nid] = (float) s; 

    }else{
        conf_vector[nid] = s; 
    }
       
}

uint SNP_model::get_spikes (uint nid)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) load_to_cpu();
    //////////////////////////////////////////////////////
    int spikes;
    if(using_lib){
        spikes = (uint) f_conf_vector[nid];
    }else{
        spikes = (uint) conf_vector[nid];
    }
    return spikes;
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model::add_rule (uint nid, uint e_n, uint e_i, uint c, uint p, uint d) 
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    assert(!done_rules);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    if (rule_index[nid+1] == -1) // first rule in neuron
        rule_index[nid+1] = rule_index[nid] + 1; 
    else   // keep accumulation
        rule_index[nid+1] = rule_index[nid+1] + 1;

    uint rid = rule_index[nid+1]-1;

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
    rules.d[rid]  = d;
    rules.nid[rid]= nid;
}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void SNP_model::add_synapse (uint i, uint j) 
{
    //////////////////////////////////////////////////////
    // ensure parameters within limits
    assert(i < n && j < n);
    // ensure all rules have been introduced already
    // assert(rule_index[n]==m);
    // SNP does not allow self-synapses
    assert(i!=j);
    done_rules = true; // from now on, no more rules can be added
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    include_synapse(i,j);
}



bool SNP_model::transition_step (int i)
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (!gpu_updated) load_to_gpu();
    //////////////////////////////////////////////////////
    if(step==0 && verbosity_lv >= 3){
        print_transition_matrix();
        print_conf_vector();
    }
    cpu_updated = false;
    bool calc_next = false;

    calc_spiking_vector();
    if(verbosity_lv >= 3){
        print_spiking_vector();
        if(delays_vector != NULL){
            print_delays_vector();
        }
        
    }
    calc_next = (i>0) || ((i==-1) && check_next_trans());
    
    if(calc_next){
        if(verbosity_lv >= 2){
            printf("\n\nstep #%d",step);
            printf("\n---------------------------------------\n");
        }

        calc_transition();
        if(i==1){
            load_to_cpu();
        }
        if(verbosity_lv >= 2){
            print_conf_vector();
        }
        step++;
    }else{
        if(verbosity_lv==1){

            /*if i counter is active, we want the exact configuration vector as it was computed after i number of steps. 
            Keep in mind that calc_spiking_vector() modifies conf_vector, so after check_next_trans(), conf_vector would have changed.
            By setting cpu_updated flag, we can bypass this so that print_conf_vector() wont find it necessary to load_to_cpu(),
            and in turn, it wont print the latest conf_vector calculated, but the one after i steps are performed */
            if(i==0){
                cpu_updated = true;
            }
            print_conf_vector();
        }else{
            cpu_updated = true;
        }
    }

    return calc_next; 
}

void SNP_model::load_to_gpu () 
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated) return;
    gpu_updated = true;
    //////////////////////////////////////////////////////

    if(using_lib){
        cudaMemcpy(df_conf_vector,   f_conf_vector,    sizeof(float)*n,   cudaMemcpyHostToDevice);
    }else{
        cudaMemcpy(d_conf_vector,   conf_vector,    sizeof(uint)*n,   cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_rule_index,    rule_index,     sizeof(uint)*(n+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.Ei,      rules.Ei,       sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.En,      rules.En,       sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.c,       rules.c,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.p,       rules.p,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.d,       rules.d,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.nid,     rules.nid,      sizeof(uint)*m,     cudaMemcpyHostToDevice);

    load_transition_matrix();
}

void SNP_model::load_to_cpu ()
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (cpu_updated) return;
    cpu_updated = true;
    //////////////////////////////////////////////////////
    
    if(using_lib){
        cudaMemcpy(f_conf_vector, df_conf_vector, sizeof(float)*n, cudaMemcpyDeviceToHost);

    }else{
        cudaMemcpy(conf_vector, d_conf_vector, sizeof(uint)*n, cudaMemcpyDeviceToHost);
    }
}

void SNP_model::compute(int i){
    float time;
    cudaEvent_t start, stop;    
    if(count_time && repetitions == -1){
        cuda_check( cudaEventCreate(&start) );
        cuda_check( cudaEventCreate(&stop) );
        cuda_check( cudaEventRecord(start, 0) );
    }
    while(transition_step(i)){
        if(i>0){
            i--;
        }
    };
  
    if(get_mem_info && this->repetitions == -1){
        size_t free_bytes;
        size_t total_bytes;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        double used_mem = (total_bytes - free_bytes)/1024/1024;
        printf("Used memory: %f MB\n", used_mem);
    }
    

    if(count_time && this->repetitions == -1){
        cuda_check( cudaEventRecord(stop, 0) );
        cuda_check( cudaEventSynchronize(stop) );
        cuda_check( cudaEventElapsedTime(&time, start, stop) );

        printf("Execution time: %3.1f ms\n", time);
    }

    if(this->outfile != NULL){
        write_to_file();
    }
}