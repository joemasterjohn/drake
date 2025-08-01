/**
 * Performs the checkout step for drake (cloning into WORKSPACE/'src') and
 * drake-ci (cloning into WORKSPACE/'ci').
 * 
 * @param ciSha the commit SHA or branch name to use for drake-ci
 * @param drakeSha the commit SHA or branch name to use for drake; if none is
 *                 given, uses the current branch or pull request
 * @return the scmVars object from the drake checkout
 */
def checkout(String ciSha = 'main', String drakeSha = null) {
  def scmVars = null
  retry(4) {
    if (drakeSha) {
      scmVars = checkout([$class: 'GitSCM',
        branches: [[name: drakeSha]],
        extensions: [[$class: 'AuthorInChangelog'],
          [$class: 'CloneOption', honorRefspec: true, noTags: true],
          [$class: 'RelativeTargetDirectory', relativeTargetDir: 'src'],
          [$class: 'LocalBranch', localBranch: 'master']],
        userRemoteConfigs: [[
          credentialsId: 'ad794d10-9bc8-4a7a-a2f3-998af802cab0',
          name: 'origin',
          refspec: '+refs/heads/*:refs/remotes/origin/* ' +
            '+refs/pull/*:refs/remotes/origin/pr/*',
          url: 'git@github.com:RobotLocomotion/drake.git']]])
    }
    else {
      dir("${env.WORKSPACE}/src") {
        scmVars = checkout scm
      }
    }
  }
  retry(4) {
    checkout([$class: 'GitSCM',
      branches: [[name: ciSha]],
      extensions: [[$class: 'AuthorInChangelog'],
        [$class: 'CloneOption', honorRefspec: true, noTags: true],
        [$class: 'RelativeTargetDirectory', relativeTargetDir: 'ci'],
        [$class: 'LocalBranch', localBranch: 'main']],
      userRemoteConfigs: [[
        credentialsId: 'ad794d10-9bc8-4a7a-a2f3-998af802cab0',
        name: 'origin',
        refspec: '+refs/heads/*:refs/remotes/origin/* ' +
          '+refs/pull/*:refs/remotes/origin/pr/*',
        url: 'git@github.com:RobotLocomotion/drake-ci.git']]])
  }
  return scmVars
}

/**
 * Performs the main build step by calling into a drake-ci driver script
 * with the necessary credentials and environment variables.
 *
 * @param scmVars the scmVars object from drake (obtained via checkout)
 * @param stagingReleaseVersion for staging jobs, the value of the environment
 *                              variable DRAKE_VERSION used by drake-ci
 */
def doMainBuild(Map scmVars, String stagingReleaseVersion = null) {
  if (env.JOB_NAME.contains("cache-server-health-check")) {
    echo "Checking the cache server:"
    sh "${env.WORKSPACE}/ci/cache_server/health_check.bash"
  }
  else {
    withCredentials([
      sshUserPrivateKey(credentialsId: 'ad794d10-9bc8-4a7a-a2f3-998af802cab0',
        keyFileVariable: 'SSH_PRIVATE_KEY_FILE'),
      string(credentialsId: 'e21b9517-8aa7-419e-8f25-19cd42e10f68',
        variable: 'DOCKER_USERNAME'),
      file(credentialsId: '912dd413-d419-4760-b7ab-c132ab9e7c5e',
        variable: 'DOCKER_PASSWORD_FILE')
    ]) {
      def environment = ["GIT_COMMIT=${scmVars.GIT_COMMIT}"]
      if (stagingReleaseVersion) {
        environment += "DRAKE_VERSION=${stagingReleaseVersion}"
      }
      withEnv(environment) {
        sh "${env.WORKSPACE}/ci/ctest_driver_script_wrapper.bash"
      }
    }
  }
}

/**
 * Sends an email to Drake developers when a build fails or is unstable.
 */
def emailFailureResults() {
  if (fileExists('RESULT')) {
    currentBuild.result = readFile 'RESULT'
    if (currentBuild.result == 'FAILURE' ||
        currentBuild.result == 'UNSTABLE') {
      def subject = 'Build failed in Jenkins'
      if (currentBuild.result == 'UNSTABLE') {
        subject = 'Jenkins build is unstable'
      }
      emailext (
        subject: "${subject}: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
        body: "See <${env.BUILD_URL}display/redirect?page=changes> " +
          "and <${env.BUILD_URL}changes>",
        to: '$DEFAULT_RECIPIENTS',
      )
    }
  }
}

/**
 * Deletes the workspace and tmp directories, for use at the end of a build.
 */
def cleanWorkspace() {
  dir(env.WORKSPACE) {
    deleteDir()
  }
  dir("${env.WORKSPACE}@tmp") {
    deleteDir()
  }
}

/**
 * Provides links to CDash to view the results of the build.
 */
def addCDashBadge() {
  if (fileExists('CDASH')) {
    def cDashUrl = readFile 'CDASH'
    addBadge icon: '/userContent/cdash.png',
      link: cDashUrl, text: 'View in CDash'
    addSummary icon: '/userContent/cdash.png',
      link: cDashUrl, text: 'View in CDash'
  }
}

// This must be present in order for this script to be consumed by
// Jenkinsfiles when using 'load'.
return this
