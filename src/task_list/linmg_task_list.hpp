#ifndef TASK_LIST_LINMG_TASK_LIST_HPP_
#define TASK_LIST_LINMG_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linmg_task_list.hpp
//! \brief define LinearMGTaskList

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
//! \class LinearMGBoundaryTaskList
//! \brief data and function definitions for LinearMGBoundaryTaskList derived class

class LinearMGBoundaryTaskList : public TaskList {
 public:
  LinearMGBoundaryTaskList(ParameterInput *pin, Mesh *pm);

  // functions
  TaskStatus ClearLinearMGBoundary(MeshBlock *pmb, int stage);
  TaskStatus SendLinearMGBoundary(MeshBlock *pmb, int stage);
  TaskStatus ReceiveLinearMGBoundary(MeshBlock *pmb, int stage);
  TaskStatus SetLinearMGBoundary(MeshBlock *pmb, int stage);
  TaskStatus ProlongateLinearMGBoundary(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);

 private:
  void AddTask(const TaskID& id, const TaskID& dep) override;
  void StartupTaskList(MeshBlock *pmb, int stage) override;
};


//----------------------------------------------------------------------------------------
//! 64-bit integers with "1" in different bit positions used to ID  each hydro task.
namespace LinearMGBoundaryTaskNames {
const TaskID NONE(0);
const TaskID CLEAR_DATA(1);

const TaskID SEND_DATA_BND(2);
const TaskID RECV_DATA_BND(3);
const TaskID SETB_DATA_BND(4);
const TaskID PROLONG_DATA_BND(5);
const TaskID DATA_PHYS_BND(6);
}
#endif // TASK_LIST_LINMG_TASK_LIST_HPP_
