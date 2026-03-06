//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file nr_task_list.cpp
//! \brief functions for NewtonRaphsonTaskList class

// C headers

// C++ headers
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../newton_raphson/Newton_Raphson.hpp"
#include "nr_task_list.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

using namespace NewtonRaphsonTaskNames; // NOLINT (build/namespace)

NewtonRaphsonTaskList::NewtonRaphsonTaskList(NewtonRaphsonDriver *pmd, Mode mode)
    : ntasks(0),
      task_list_{},
      pmy_driver_(pmd),
      mode_(mode) {
  if (mode_ == Mode::coeff) {
    AddNewtonRaphsonTask(CALC_COEFF, NONE);
  } else {
    AddNewtonRaphsonTask(APPLY_CORR, NONE);
    AddNewtonRaphsonTask(START_RECV, APPLY_CORR);
    AddNewtonRaphsonTask(SEND_BND, START_RECV);
    AddNewtonRaphsonTask(RECV_BND, SEND_BND);
    AddNewtonRaphsonTask(SET_BND, RECV_BND);
    if (pmy_driver_->pmy_mesh_->multilevel) {
      AddNewtonRaphsonTask(PROLONG, SET_BND);
      AddNewtonRaphsonTask(PHYS_BND, PROLONG);
    } else {
      AddNewtonRaphsonTask(PHYS_BND, SET_BND);
    }
    AddNewtonRaphsonTask(CLEAR_BND, PHYS_BND);
  }
}

void NewtonRaphsonTaskList::DoTaskListOneStage(int stage) {
  NewtonRaphsonDriver *pmd = pmy_driver_;
  int nthreads = pmd->nthreads_;
  int nnr_left = static_cast<int>(pmd->vnr_.size());

#pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
  for (auto itr = pmd->vnr_.begin(); itr < pmd->vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    pnr->ts_.Reset(ntasks);
  }

  while (nnr_left > 0) {
#pragma omp parallel for reduction(- : nnr_left) num_threads(nthreads) schedule(dynamic,1)
    for (auto itr = pmd->vnr_.begin(); itr < pmd->vnr_.end(); itr++) {
      NewtonRaphson *pnr = *itr;
      if (DoAllAvailableTasks(pnr, pnr->ts_) == TaskListStatus::complete) nnr_left--;
    }
  }
  return;
}

TaskListStatus NewtonRaphsonTaskList::DoAllAvailableTasks(NewtonRaphson *pnr,
                                                          TaskStates &ts) {
  int skip=0;
  TaskStatus ret;

  if (ts.num_tasks_left==0) return TaskListStatus::nothing_to_do;

  for (int i=ts.indx_first_task; i<ntasks; i++) {
    NRTask &taski=task_list_[i];

    if (ts.finished_tasks.IsUnfinished(taski.task_id)) {
      if (ts.finished_tasks.CheckDependencies(taski.dependency)) {
        ret = (this->*taski.TaskFunc)(pnr);
        if (ret != TaskStatus::fail) {
          ts.num_tasks_left--;
          ts.finished_tasks.SetFinished(taski.task_id);
          if (skip==0) ts.indx_first_task++;
          if (ts.num_tasks_left==0) return TaskListStatus::complete;
          if (ret==TaskStatus::next) continue;
          return TaskListStatus::running;
        }
      }
      skip++; // increment number of tasks processed
    } else if (skip==0) {
      ts.indx_first_task++;
    }
  }
  return TaskListStatus::stuck;
}

void NewtonRaphsonTaskList::AddNewtonRaphsonTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id = id;
  task_list_[ntasks].dependency = dep;

  if (id == CALC_COEFF) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::CalculateCoefficients;
  } else if (id == APPLY_CORR) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::ApplyCorrection;
  } else if (id == START_RECV) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::StartBoundary;
  } else if (id == SEND_BND) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::SendBoundary;
  } else if (id == RECV_BND) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::ReceiveBoundary;
  } else if (id == SET_BND) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::SetBoundary;
  } else if (id == PROLONG) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::ProlongateBoundary;
  } else if (id == PHYS_BND) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::PhysicalBoundary;
  } else if (id == CLEAR_BND) {
    task_list_[ntasks].TaskFunc = &NewtonRaphsonTaskList::ClearBoundary;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in NewtonRaphsonTaskList::AddNewtonRaphsonTask" << std::endl
        << "Unknown task ID." << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

TaskStatus NewtonRaphsonTaskList::CalculateCoefficients(NewtonRaphson *pnr) {
  pnr->CalculateCoefficientsTask(pmy_driver_->dt_);
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::ApplyCorrection(NewtonRaphson *pnr) {
  pnr->ApplyCorrectionTask();
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::StartBoundary(NewtonRaphson *pnr) {
  pnr->StartBoundary();
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::SendBoundary(NewtonRaphson *pnr) {
  pnr->SendBoundary();
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::ReceiveBoundary(NewtonRaphson *pnr) {
  if (!pnr->ReceiveBoundary()) return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::SetBoundary(NewtonRaphson *pnr) {
  pnr->SetBoundary();
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::ProlongateBoundary(NewtonRaphson *pnr) {
  pnr->ProlongateBoundary(pmy_driver_->dt_);
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::PhysicalBoundary(NewtonRaphson *pnr) {
  pnr->ApplyPhysicalBoundary();
  return TaskStatus::success;
}

TaskStatus NewtonRaphsonTaskList::ClearBoundary(NewtonRaphson *pnr) {
  pnr->ClearBoundary();
  return TaskStatus::success;
}
