//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rad_fld_task_list.cpp
//! \brief function implementation for FLDBoundaryTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../rad_fld/rad_fld.hpp"
#include "rad_fld_task_list.hpp"
#include "task_list.hpp"

//----------------------------------------------------------------------------------------
//! FLDBoundaryTaskList constructor

FLDBoundaryTaskList::FLDBoundaryTaskList(ParameterInput *pin, Mesh *pm) {
  // Now assemble list of tasks for each stage of time integrator
  {using namespace FLDBoundaryTaskNames; // NOLINT (build/namespace)
    AddTask(SEND_FLD_BND,NONE);
    AddTask(RECV_FLD_BND,NONE);
    AddTask(SETB_FLD_BND,(RECV_FLD_BND|SEND_FLD_BND));
    if (pm->multilevel) {
      AddTask(PROLONG_FLD_BND,SETB_FLD_BND);
      AddTask(FLD_PHYS_BND,PROLONG_FLD_BND);
    } else {
      AddTask(FLD_PHYS_BND,SETB_FLD_BND);
    }
    AddTask(CLEAR_FLD, FLD_PHYS_BND);
    AddTask(UPD_OPA,FLD_PHYS_BND);
  } // end of using namespace block
}

//----------------------------------------------------------------------------------------
//! \fn void FLDBoundaryTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void FLDBoundaryTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace FLDBoundaryTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_FLD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::ClearFLDBoundary);
  } else if (id == SEND_FLD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::SendFLDBoundary);
  } else if (id == RECV_FLD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::ReceiveFLDBoundary);
  } else if (id == SETB_FLD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::SetFLDBoundary);
  } else if (id == PROLONG_FLD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::ProlongateFLDBoundary);
  } else if (id == FLD_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::PhysicalBoundary);
  } else if (id == UPD_OPA) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FLDBoundaryTaskList::UpdateOpacity);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in FLDBoundaryTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

void FLDBoundaryTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  pmb->prfld->mgfldbvar.StartReceiving(BoundaryCommSubset::all);
  return;
}

TaskStatus FLDBoundaryTaskList::ClearFLDBoundary(MeshBlock *pmb,
                                                                 int stage) {
  pmb->prfld->mgfldbvar.ClearBoundary(BoundaryCommSubset::all);
  return TaskStatus::success;
}

TaskStatus FLDBoundaryTaskList::SendFLDBoundary(MeshBlock *pmb,
                                                                int stage) {
  pmb->prfld->mgfldbvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus FLDBoundaryTaskList::ReceiveFLDBoundary(MeshBlock *pmb,
                                                                   int stage) {
  bool ret = pmb->prfld->mgfldbvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus FLDBoundaryTaskList::SetFLDBoundary(MeshBlock *pmb,
                                                               int stage) {
  pmb->prfld->mgfldbvar.SetBoundaries();
  return TaskStatus::success;
}

TaskStatus FLDBoundaryTaskList::ProlongateFLDBoundary(MeshBlock *pmb,
                                                                      int stage) {
  pmb->pbval->ProlongateBoundariesPostMG(&(pmb->prfld->mgfldbvar));
  return TaskStatus::success;
}

TaskStatus FLDBoundaryTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  pmb->prfld->mgfldbvar.ExpandPhysicalBoundaries();
  return TaskStatus::next;
}

TaskStatus FLDBoundaryTaskList::UpdateOpacity(MeshBlock *pmb, int stage) {
  pmb->prfld->UpdateOpacity(pmb, pmb->prfld->u, pmb->phydro->w);
  return TaskStatus::success;
}
