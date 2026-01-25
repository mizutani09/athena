#ifndef TASK_LIST_NR_TASK_LIST_HPP_
#define TASK_LIST_NR_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file nr_task_list.hpp
//! \brief define NewtonRaphsonTaskList class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "task_list.hpp"

// forward declarations
class NewtonRaphson;
class NewtonRaphsonDriver;

//----------------------------------------------------------------------------------------
//! 64-bit integers with "1" in different bit positions used to ID each task.

namespace NewtonRaphsonTaskNames {
const TaskID NONE(0);
const TaskID CALC_COEFF(1);
const TaskID APPLY_CORR(2);
const TaskID START_RECV(3);
const TaskID SEND_BND(4);
const TaskID RECV_BND(5);
const TaskID SET_BND(6);
const TaskID PROLONG(7);
const TaskID PHYS_BND(8);
const TaskID CLEAR_BND(9);
} // namespace NewtonRaphsonTaskNames

//----------------------------------------------------------------------------------------
//! \struct NRTask
//! \brief data and function pointer for an individual NewtonRaphson task

struct NRTask {
  TaskID task_id;
  TaskID dependency;
  TaskStatus (NewtonRaphsonTaskList::*TaskFunc)(NewtonRaphson*);
};

//----------------------------------------------------------------------------------------
//! \class NewtonRaphsonTaskList
//! \brief data and function definitions for NewtonRaphsonTaskList class

class NewtonRaphsonTaskList {
 public:
  enum class Mode {coeff, post};
  NewtonRaphsonTaskList(NewtonRaphsonDriver *pmd, Mode mode);

  void DoTaskListOneStage(int stage);
  TaskListStatus DoAllAvailableTasks(NewtonRaphson *pnr, TaskStates &ts);

  TaskStatus CalculateCoefficients(NewtonRaphson *pnr);
  TaskStatus ApplyCorrection(NewtonRaphson *pnr);
  TaskStatus StartBoundary(NewtonRaphson *pnr);
  TaskStatus SendBoundary(NewtonRaphson *pnr);
  TaskStatus ReceiveBoundary(NewtonRaphson *pnr);
  TaskStatus SetBoundary(NewtonRaphson *pnr);
  TaskStatus ProlongateBoundary(NewtonRaphson *pnr);
  TaskStatus PhysicalBoundary(NewtonRaphson *pnr);
  TaskStatus ClearBoundary(NewtonRaphson *pnr);

 private:
  void AddNewtonRaphsonTask(const TaskID& id, const TaskID& dep);

  int ntasks;
  NRTask task_list_[64*TaskID::kNField_];

  NewtonRaphsonDriver *pmy_driver_;
  Mode mode_;
};

#endif // TASK_LIST_NR_TASK_LIST_HPP_
