#ifndef TASK_LIST_RAD_FLD_TASK_LIST_HPP_
#define TASK_LIST_RAD_FLD_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rad_fld_task_list.hpp
//! \brief define FLDBoundaryTaskList

// C headers

// C++ headers
#include <cstdint>      // std::uint64_t

// Athena++ headers
#include "../athena.hpp"
#include "task_list.hpp"

// forward declarations
class Mesh;
class MeshBlock;

//----------------------------------------------------------------------------------------
//! \class FLDBoundaryTaskList
//! \brief data and function definitions for FLDBoundaryTaskList derived class

class FLDBoundaryTaskList : public TaskList {
 public:
  FLDBoundaryTaskList(ParameterInput *pin, Mesh *pm);

  // functions
  TaskStatus ClearFLDBoundary(MeshBlock *pmb, int stage);
  TaskStatus SendFLDBoundary(MeshBlock *pmb, int stage);
  TaskStatus ReceiveFLDBoundary(MeshBlock *pmb, int stage);
  TaskStatus SetFLDBoundary(MeshBlock *pmb, int stage);
  TaskStatus ProlongateFLDBoundary(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);

 private:
  void AddTask(const TaskID& id, const TaskID& dep) override;
  void StartupTaskList(MeshBlock *pmb, int stage) override;
};


//----------------------------------------------------------------------------------------
//! 64-bit integers with "1" in different bit positions used to ID  each hydro task.
namespace FLDBoundaryTaskNames {
const TaskID NONE(0);
const TaskID CLEAR_FLD(1);

const TaskID SEND_FLD_BND(2);
const TaskID RECV_FLD_BND(3);
const TaskID SETB_FLD_BND(4);
const TaskID PROLONG_FLD_BND(5);
const TaskID FLD_PHYS_BND(6);
}
#endif // TASK_LIST_RAD_FLD_TASK_LIST_HPP_
