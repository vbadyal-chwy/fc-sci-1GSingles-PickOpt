def getGradleTasks(branch) {
    if (branch == 'master'|| branch== 'main') {
        return 'final dockerPushImage'
    } else {
        return 'devSnapshot dockerPushImage'
    }
}

pipeline {
    agent {
        label 'amzlnx2'
    }

    tools {
        jdk 'openjdk-11.0.2'
        // used for terraform validate stage
        terraform 'terraform-1.0.2'
    }

    stages {
        // stage('terraform validate') {
        //     steps {
        //         dir("terraform") {
        //             sh "terraform init -backend=false"
        //             sh "terraform validate"
        //         }
        //     }
        // }
        stage('gradle build') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'jenkins-github-userpass',
                                                  passwordVariable: 'GRGIT_PASS',
                                                  usernameVariable: 'GRGIT_USER')]) {
                    script {
                            def gradleTasks = getGradleTasks("${env.BRANCH_NAME}")
                            sh """#!/bin/bash
                                ${ecrLogin()}
                                ./gradlew ${gradleTasks}
                            """
                    }
                }
            }
        }
    }
    post {
        success {
            script {
                def version = readFile "version.txt"
                // Change to description if you want it below the title
                currentBuild.displayName = "#${BUILD_NUMBER}: ${version}"
            }
        }
    }
}
