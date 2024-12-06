.. ## Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _macmini-label:

==========================================
Adding an Additional User to Serac MacMini
==========================================

This page assumes you are a Serac developer who requires access to the team's shared MacMini. This machine
tests Serac Mac builds on a regular basis via cron. If you have any questions, reach out to either
`Brandon Talamini <talamini1@llnl.gov>`_, `Alex Chapman <chapman39@llnl.gov>`_, or LivIT. The following
are steps to guide you to gaining access on the machine to the point you're able to build Serac.

1. **Add User**

Without a MyPass, you can still log in using your LLNL username and AD password. Do this first to setup an account on the machine.
You won't be able to do much, since you do not have access to FileVault, and you are not an admin... yet.

2. **MyPass**

Then, acquire a new MyPass with a USB-C port dedicated to this machine. This will grant you access to FileVault.
Contact LivIT directly to setup an appointment and request one.

3. **EARS Admin Request**

Next, request admin access to the machine by visiting either ServiceNow or the `EARS website <https://ears.llnl.gov/dashboard>`_.

4. **Two Logins**

Once you have a MyPass and you have admin rights, try to log in again. There should be two passwords required to log in. The first one
is for your MyPass (assuming it's connected to the machine) and the other is for the account login.

5. **Download and setup Brew**

Visit `Brew's website <https://brew.sh/>`_ to install and setup Brew. This is required to install some of Serac's third-party libraries
(TPLs).

6. **Add New SSH Key to GitHub**

Next step setting up a new SSH Key to your GitHub account so that you're able to clone the Serac repo. In case you do not know
how to do this, instructions can be found on
`GitHub's website <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_.

7. **Install Serac**

You're now able to clone Serac and get started with the installation process. Further instructions for doing so are currently on 
the `quickstart page <https://serac.readthedocs.io/en/latest/sphinx/quickstart.html#quickstart-label>`_ of the Serac documentation.
