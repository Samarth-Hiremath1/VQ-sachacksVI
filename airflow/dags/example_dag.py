"""
Example DAG for AI Communication Coaching Platform
This is a placeholder DAG that will be replaced with actual processing workflows
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'coaching-platform',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def hello_world():
    """Simple hello world function"""
    print("Hello from AI Communication Coaching Platform!")
    return "success"

# Define the DAG
dag = DAG(
    'example_coaching_dag',
    default_args=default_args,
    description='Example DAG for coaching platform',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['example', 'coaching'],
)

# Define tasks
hello_task = PythonOperator(
    task_id='hello_world',
    python_callable=hello_world,
    dag=dag,
)

health_check = BashOperator(
    task_id='health_check',
    bash_command='echo "System health check completed"',
    dag=dag,
)

# Set task dependencies
hello_task >> health_check