//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linmg_task_list.cpp
//! \brief function implementation for LinearMGTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../gravity/gravity.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../newton_raphson/Newton_Raphson.hpp"
#include "linmg_task_list.hpp"
#include "task_list.hpp"

//----------------------------------------------------------------------------------------
//! LinearMGBoundaryTaskList constructor

LinearMGBoundaryTaskList::LinearMGBoundaryTaskList(ParameterInput *pin, Mesh *pm) {
  {using namespace LinearMGBoundaryTaskNames; // NOLINT (build/namespace)
    AddTask(SEND_DATA_BND,NONE);
    AddTask(RECV_DATA_BND,NONE);
    AddTask(SETB_DATA_BND,(RECV_DATA_BND|SEND_DATA_BND));
    if (pm->multilevel) {
      AddTask(PROLONG_DATA_BND,SETB_DATA_BND);
      AddTask(DATA_PHYS_BND,PROLONG_DATA_BND);
    } else {
      AddTask(DATA_PHYS_BND,SETB_DATA_BND);
    }
    AddTask(CLEAR_DATA, DATA_PHYS_BND);
  } // end of using namespace block
}

//----------------------------------------------------------------------------------------
//! \fn void LinearMGBoundaryTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void LinearMGBoundaryTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace LinearMGBoundaryTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_DATA) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&LinearMGBoundaryTaskList::ClearLinearMGBoundary);
  } else if (id == SEND_DATA_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&LinearMGBoundaryTaskList::SendLinearMGBoundary);
  } else if (id == RECV_DATA_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&LinearMGBoundaryTaskList::ReceiveLinearMGBoundary);
  } else if (id == SETB_DATA_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&LinearMGBoundaryTaskList::SetLinearMGBoundary);
  } else if (id == PROLONG_DATA_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&LinearMGBoundaryTaskList::ProlongateLinearMGBoundary);
  } else if (id == DATA_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&LinearMGBoundaryTaskList::PhysicalBoundary);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in LinearMGBoundaryTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

void LinearMGBoundaryTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  // std::cout << "In LinearMGBoundaryTaskList::StartupTaskList" << std::endl;
  pmb->pnr->delta_bvar.StartReceiving(BoundaryCommSubset::all);
  // std::cout << "LinearMGBoundaryTaskList startup tasks done at " << Globals::my_rank << " gid "
  //           << pmb->gid << std::endl;
  return;
}

TaskStatus LinearMGBoundaryTaskList::ClearLinearMGBoundary(MeshBlock *pmb, int stage) {
  // std::cout << "Clearing LinearMG boundary buffers." << std::endl;
  pmb->pnr->delta_bvar.ClearBoundary(BoundaryCommSubset::all);
  // std::cout << "LinearMG boundary buffers cleared." << std::endl;
  return TaskStatus::success;
}

TaskStatus LinearMGBoundaryTaskList::SendLinearMGBoundary(MeshBlock *pmb, int stage) {
  // std::cout << "Sending LinearMG boundary buffers." << std::endl;
  pmb->pnr->delta_bvar.SendBoundaryBuffers();
  // std::cout << "LinearMG boundary buffers sent." << std::endl;
  return TaskStatus::success;
}

TaskStatus LinearMGBoundaryTaskList::ReceiveLinearMGBoundary(MeshBlock *pmb,
                                                               int stage) {
  // std::cout << "Receiving LinearMG boundary buffers." << std::endl;
  bool ret = pmb->pnr->delta_bvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  // std::cout << "LinearMG boundary buffers received." << std::endl;
  return TaskStatus::success;
}

TaskStatus LinearMGBoundaryTaskList::SetLinearMGBoundary(MeshBlock *pmb, int stage) {
  // std::cout << "Setting LinearMG boundary." << std::endl;
  pmb->pnr->delta_bvar.SetBoundaries();
  // std::cout << "LinearMG boundary set." <<std::endl;
  return TaskStatus::success;
}

TaskStatus LinearMGBoundaryTaskList::ProlongateLinearMGBoundary(MeshBlock *pmb,
                                                              int stage) {
  // std::cout << "Prolongate LinearMG." << std::endl;
  pmb->pbval->ProlongateBoundariesPostMG(&(pmb->pnr->delta_bvar));
  // std::cout << "LinearMG Prolongated" << std::endl;
  return TaskStatus::success;
}

TaskStatus LinearMGBoundaryTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  // std::cout << "Apply PhysicalBoundary" << std::endl;
  pmb->pnr->delta_bvar.ExpandPhysicalBoundaries();
  // std::cout << "End PhysicalBoundary" << std::endl;
  return TaskStatus::next;
}
